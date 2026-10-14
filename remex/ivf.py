"""Coarse IVF index over the Matryoshka tier.

Partitions a compressed corpus into ``2**n_bits`` cells via a
**data-oblivious** hash — no k-means, no training, no fitting. Visiting
only ``nprobe`` cells per query trades recall for stage-1 latency in
two-stage retrieval. The trade-off is benchmarked in
``bench/specter2_eval.py``.

Two hash modes are provided:

- ``'lsh'``: random-hyperplane LSH (a.k.a. SimHash). ``n_bits`` random
  Gaussian hyperplanes, fixed by ``(d, n_bits, seed)``. Cell ID is the
  sign pattern of projections onto those hyperplanes. Pure
  data-oblivious — works on any embedding distribution.

- ``'rotated_prefix'``: sign of the first ``n_bits`` post-rotation
  coordinates. Free given the existing remex rotation matrix — these
  bits are already the MSBs of the encoded indices. Cell balance
  depends on whether rotated coordinates are approximately i.i.d.
  Gaussian (verified for SPECTER2 in ``bench/specter2_eval.py``).

Multi-probe is by Hamming distance from the query's hash code: the
``nprobe`` cells with the lowest Hamming distance to ``q_hash`` are
visited (ties broken by cell ID). Setting ``nprobe`` to ``n_cells``
recovers a flat scan over the candidate set.

Memory accounting (excluding the underlying ``CompressedVectors`` /
``PackedVectors`` storage):
  - ``cell_ids`` (n,): 2 bytes per vector  (``uint16``, n_bits ≤ 16)
  - ``sorted_idx`` (n,): 8 bytes per vector (``int64``)
  - ``cell_offsets`` (2**n_bits + 1,): 8 bytes per cell (``int64``)
  - ``hyperplanes`` (n_bits, d): 4 * n_bits * d bytes (LSH only)

For 100M vectors at ``n_bits=12``: ~960 MB index overhead vs ~9.6 GB
coarse memory — under 10% surcharge for ~10–100x stage-1 speedup at
moderate ``nprobe``.

Honest caveats — both hash modes are *coarse* and lose recall vs flat
scan. They make sense only when stage-1 latency is the bottleneck and
some recall@K loss is acceptable. The benchmark in
``bench/specter2_eval.py`` reports the latency–recall Pareto.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from remex.core import (
    CompressedVectors, PackedVectors, Quantizer, _apply_norms,
)


_VALID_MODES = ("lsh", "rotated_prefix")


def _popcount32(x: np.ndarray) -> np.ndarray:
    """Vectorized SWAR popcount on a uint32 ndarray."""
    x = x.astype(np.uint32, copy=True)
    x = x - ((x >> 1) & np.uint32(0x55555555))
    x = (x & np.uint32(0x33333333)) + ((x >> 2) & np.uint32(0x33333333))
    x = (x + (x >> 4)) & np.uint32(0x0F0F0F0F)
    x = (x * np.uint32(0x01010101)) >> np.uint32(24)
    return x.astype(np.uint8)


class IVFCoarseIndex:
    """Inverted-file index over the coarse Matryoshka tier.

    Args:
        quantizer: ``Quantizer`` used to encode the corpus.
        compressed: Encoded corpus (``CompressedVectors`` or
            ``PackedVectors``) — stored by reference; not copied.
        n_bits: Number of hash bits. The index has ``2**n_bits`` cells.
            Must be 1..16. For uniform LSH/sign-prefix hashes the
            average cell size is roughly ``n / 2**n_bits``.
        mode: ``'lsh'`` or ``'rotated_prefix'``.
        seed: Seed for the LSH hyperplane RNG. Ignored for
            ``rotated_prefix`` (which derives its hash from the
            quantizer's rotation matrix). Defaults to 0 so the index is
            deterministic from ``(quantizer, n_bits, mode)``.

    Attributes:
        n_cells: ``2**n_bits``.
        cell_ids: ``(n,)`` uint16 cell ID per corpus vector.
        sorted_idx: ``(n,)`` int64 corpus row indices sorted by cell.
        cell_offsets: ``(n_cells + 1,)`` int64 CSR-style offsets into
            ``sorted_idx``. Cell ``c`` occupies
            ``sorted_idx[cell_offsets[c]:cell_offsets[c+1]]``.
        hyperplanes: ``(n_bits, d)`` float32 (LSH only) — random
            hyperplanes in rotated space.
    """

    def __init__(
        self,
        quantizer: Quantizer,
        compressed,
        n_bits: int,
        mode: str = "lsh",
        seed: int = 0,
    ):
        if n_bits < 1 or n_bits > 16:
            raise ValueError(f"n_bits must be 1-16, got {n_bits}")
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES}, got {mode!r}"
            )
        if mode == "rotated_prefix" and n_bits > quantizer.d:
            raise ValueError(
                f"n_bits={n_bits} exceeds quantizer.d={quantizer.d} "
                f"for rotated_prefix mode"
            )
        if not isinstance(compressed, (CompressedVectors, PackedVectors)):
            raise TypeError(
                f"compressed must be CompressedVectors or PackedVectors, "
                f"got {type(compressed).__name__}"
            )

        self.quantizer = quantizer
        self.compressed = compressed
        self.n_bits = int(n_bits)
        self.n_cells = 1 << int(n_bits)
        self.mode = mode
        self.seed = int(seed)
        self.n = compressed.n
        self.d = quantizer.d

        if mode == "lsh":
            rng = np.random.default_rng(seed)
            self.hyperplanes = rng.standard_normal(
                (self.n_bits, self.d)
            ).astype(np.float32)
        else:
            self.hyperplanes = None

        self.cell_ids = self._compute_cell_ids()
        self.sorted_idx, self.cell_offsets = self._build_inverted_lists()

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _shift_to_msb(self) -> int:
        """Right-shift to extract the MSB (= 1-bit Lloyd-Max code)."""
        return self.quantizer.bits - 1

    def _indices_chunk(self, start: int, end: int) -> np.ndarray:
        if isinstance(self.compressed, PackedVectors):
            return self.compressed.unpack_rows(start, end)
        return self.compressed.indices[start:end]

    def _pack_bits_to_cell(self, sign_bits: np.ndarray) -> np.ndarray:
        """Pack ``(n, n_bits)`` {0,1} array into ``(n,)`` uint32 cell IDs.

        Bit ``b`` of the output is taken from column ``b`` of ``sign_bits``.
        """
        n = sign_bits.shape[0]
        out = np.zeros(n, dtype=np.uint32)
        sign_bits_u32 = sign_bits.astype(np.uint32, copy=False)
        for b in range(self.n_bits):
            out |= sign_bits_u32[:, b] << np.uint32(b)
        return out

    def _compute_cell_ids(self) -> np.ndarray:
        """Compute ``(n,)`` uint16 cell ID per corpus vector."""
        if self.mode == "rotated_prefix":
            return self._cell_ids_rotated_prefix()
        return self._cell_ids_lsh()

    def _cell_ids_rotated_prefix(self) -> np.ndarray:
        """1-bit MSB extraction of the first ``n_bits`` rotated coords.

        The MSB of an n-bit Lloyd-Max code is exactly the sign of the
        rotated coordinate (see ``test_matryoshka_1bit_equals_standalone_1bit``),
        so this is a free, deterministic hash given the encoded corpus.
        """
        shift = self._shift_to_msb()
        cell_ids = np.zeros(self.n, dtype=np.uint32)
        chunk = 65_536
        for start in range(0, self.n, chunk):
            end = min(start + chunk, self.n)
            indices_chunk = self._indices_chunk(start, end)
            sign_bits = (indices_chunk[:, : self.n_bits] >> shift).astype(
                np.uint8
            )
            cell_ids[start:end] = self._pack_bits_to_cell(sign_bits)
        return cell_ids.astype(np.uint16)

    def _cell_ids_lsh(self) -> np.ndarray:
        """Random-hyperplane LSH on the 1-bit reconstruction of corpus.

        The 1-bit reconstruction in rotated space is ``c1 * sign(x_rot)``
        for a positive scalar ``c1``. Sign-of-projection is invariant to
        positive scaling, so we work directly with ``signed = ±1`` from
        the MSBs of the encoded indices.

        At query time we project the *full-precision* ``q_rot`` onto the
        same hyperplanes — sign correlation between ``H @ q_rot`` and
        ``H @ sign(x_rot)`` is high for Gaussian-like rotated coords, so
        the cell assignment is consistent enough for coarse retrieval.
        Recall is the empirical check (see ``bench/specter2_eval.py``).
        """
        shift = self._shift_to_msb()
        cell_ids = np.zeros(self.n, dtype=np.uint32)
        H_T = self.hyperplanes.T  # (d, n_bits)
        chunk = 4096
        for start in range(0, self.n, chunk):
            end = min(start + chunk, self.n)
            indices_chunk = self._indices_chunk(start, end)
            # Map MSB ∈ {0, 1} → signed ∈ {-1, +1} as float32
            msb = (indices_chunk >> shift).astype(np.float32)
            signed = 2.0 * msb - 1.0  # (chunk, d)
            proj = signed @ H_T  # (chunk, n_bits)
            sign_bits = (proj > 0).astype(np.uint8)
            cell_ids[start:end] = self._pack_bits_to_cell(sign_bits)
        return cell_ids.astype(np.uint16)

    def _build_inverted_lists(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(sorted_idx, cell_offsets)`` CSR layout for the cells."""
        order = np.argsort(self.cell_ids, kind="stable").astype(np.int64)
        sorted_cells = self.cell_ids[order]
        counts = np.bincount(sorted_cells, minlength=self.n_cells)
        offsets = np.empty(self.n_cells + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        return order, offsets

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _query_cell_from_rot(self, q_rot: np.ndarray) -> int:
        """Compute cell ID from a pre-rotated query."""
        if self.mode == "rotated_prefix":
            sign_bits = (q_rot[: self.n_bits] > 0).astype(np.uint8)
        else:
            proj = self.hyperplanes @ q_rot  # (n_bits,)
            sign_bits = (proj > 0).astype(np.uint8)
        cid = 0
        for b in range(self.n_bits):
            cid |= int(sign_bits[b]) << b
        return cid

    def query_cell(self, query: np.ndarray) -> int:
        """Compute the cell ID for a single query vector.

        The query is rotated by the quantizer's rotation matrix before
        hashing so it lives in the same frame as the corpus hash.
        """
        q = np.asarray(query, dtype=np.float32)
        q_rot = self.quantizer._rotate_query(q)
        return self._query_cell_from_rot(q_rot)

    def probe_cells(self, query_cell: int, nprobe: int) -> np.ndarray:
        """Rank cells by Hamming distance to ``query_cell`` and return the
        ``nprobe`` nearest. Ties are broken by cell ID (deterministic).
        """
        if nprobe <= 0:
            return np.empty(0, dtype=np.int64)
        if nprobe >= self.n_cells:
            return np.arange(self.n_cells, dtype=np.int64)
        all_cells = np.arange(self.n_cells, dtype=np.uint32)
        xored = all_cells ^ np.uint32(query_cell)
        hd = _popcount32(xored)
        # lexsort: primary key = hd (asc), secondary = cell ID (asc)
        order = np.lexsort((all_cells, hd))[:nprobe]
        return all_cells[order].astype(np.int64)

    def candidate_indices(self, query_cell: int, nprobe: int) -> np.ndarray:
        """Return the corpus row indices in the top-``nprobe`` cells."""
        cells = self.probe_cells(query_cell, nprobe)
        if len(cells) == 0:
            return np.empty(0, dtype=np.int64)
        if len(cells) == 1:
            c = int(cells[0])
            return self.sorted_idx[
                self.cell_offsets[c] : self.cell_offsets[c + 1]
            ]
        slices = [
            self.sorted_idx[self.cell_offsets[c] : self.cell_offsets[c + 1]]
            for c in cells
        ]
        return np.concatenate(slices)

    def candidate_count(self, query: np.ndarray, nprobe: int) -> int:
        """Total candidate corpus rows in the top-``nprobe`` cells.

        Equivalent to ``len(candidate_indices(query_cell, nprobe))`` but
        avoids materializing the index array — useful for benchmarking
        the candidate pool size at a given nprobe.
        """
        q_cell = self.query_cell(query)
        cells = self.probe_cells(q_cell, nprobe)
        if len(cells) == 0:
            return 0
        sizes = np.diff(self.cell_offsets)
        return int(sizes[cells].sum())

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_coarse(
        self,
        query: np.ndarray,
        k: int = 500,
        nprobe: int = 1,
        precision: Optional[int] = None,
        chunk_size: int = 4096,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """IVF coarse ADC: visit ``nprobe`` cells, score by ADC.

        Equivalent to ``Quantizer.search_adc`` restricted to the union
        of the ``nprobe`` cells closest (in Hamming distance) to the
        query's hash. Suitable as the stage-1 of two-stage retrieval.

        Args:
            query: ``(d,)`` query vector (raw, not rotated).
            k: Number of coarse candidates to return.
            nprobe: Cells to visit. ``1`` is fastest, ``n_cells``
                scans every cell (= flat coarse scan).
            precision: Bit precision for ADC scoring (1 to
                ``quantizer.bits``). ``None`` = full precision.
            chunk_size: Rows per ADC chunk; controls peak temp memory.

        Returns:
            ``(indices, scores)``: top-``k`` corpus indices into the
            original corpus and their approximate inner-product scores.
            Length is ``min(k, n_visited)`` where ``n_visited`` is the
            total candidate count across the visited cells.
        """
        # guards first, so an empty cell cannot return [] for codes this
        # quantizer cannot read
        self.quantizer._resolve_centroids(self.compressed, precision)
        q = np.asarray(query, dtype=np.float32)
        q_rot = self.quantizer._rotate_query(q)

        q_cell = self._query_cell_from_rot(q_rot)
        cand = self.candidate_indices(q_cell, nprobe)
        n_cand = len(cand)
        if n_cand == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
            )

        centroids = self.quantizer._resolve_centroids(
            self.compressed, precision
        )
        table = np.outer(q_rot, centroids).astype(np.float32)

        if isinstance(self.compressed, PackedVectors):
            cand_idx = self.compressed.unpack_at(cand)
        else:
            cand_idx = self.compressed.indices[cand]

        if precision is not None and precision != self.quantizer.bits:
            shift = self.quantizer.bits - precision
            cand_idx = cand_idx >> shift

        norms = self.quantizer._score_norms(self.compressed, precision)
        cand_norms = None if norms is None else norms[cand]
        bias = self.quantizer._row_bias(self.compressed, precision)
        cand_bias = None if bias is None else bias[cand]
        offset = self.quantizer._query_offset(query)
        d = self.d
        dim_idx = np.arange(d)
        scores = np.empty(n_cand, dtype=np.float32)

        for start in range(0, n_cand, chunk_size):
            end = min(start + chunk_size, n_cand)
            chunk_idx = cand_idx[start:end]
            chunk_scores = table[dim_idx, chunk_idx].sum(axis=1)
            scores[start:end] = _apply_norms(
                chunk_scores,
                None if cand_norms is None else cand_norms[start:end],
            ) + offset
            if cand_bias is not None:
                scores[start:end] += cand_bias[start:end]

        k_eff = min(k, n_cand)
        if k_eff >= n_cand:
            order = np.argsort(-scores)
        else:
            order = np.argpartition(-scores, k_eff)[:k_eff]
            order = order[np.argsort(-scores[order])]

        return cand[order], scores[order]

    def search_twostage(
        self,
        query: np.ndarray,
        k: int = 10,
        candidates: int = 500,
        nprobe: int = 1,
        coarse_precision: Optional[int] = None,
        coarse_chunk_size: int = 4096,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """IVF coarse + full-precision fine rerank.

        Mirrors ``Quantizer.search_twostage`` but replaces the flat
        coarse ADC scan with an IVF-restricted scan over ``nprobe``
        cells. Stage 2 is identical: dequantize the candidate vectors
        at full precision and rerank by exact (quantized) inner product.

        Args:
            query: ``(d,)`` query vector.
            k: Final number of results.
            candidates: Number of stage-1 coarse candidates.
            nprobe: Cells to visit in stage 1.
            coarse_precision: Bit precision for stage 1.
                Default: ``max(1, quantizer.bits - 2)``.
            coarse_chunk_size: Rows per ADC chunk in stage 1.

        Returns:
            ``(indices, scores)``: top-``k`` corpus indices and
            full-precision scores. Length is
            ``min(k, n_stage1_candidates)``.
        """
        if coarse_precision is None:
            coarse_precision = max(1, self.quantizer.bits - 2)

        coarse_idx, _ = self.search_coarse(
            query,
            k=candidates,
            nprobe=nprobe,
            precision=coarse_precision,
            chunk_size=coarse_chunk_size,
        )
        if len(coarse_idx) == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
            )

        q = np.asarray(query, dtype=np.float32)
        q_rot = self.quantizer._rotate_query(q)
        fine_centroids = self.quantizer._resolve_centroids(
            self.compressed, None
        )
        if isinstance(self.compressed, PackedVectors):
            fine_indices = self.compressed.unpack_at(coarse_idx)
        else:
            fine_indices = self.compressed.indices[coarse_idx]
        X_hat_cand = fine_centroids[fine_indices]

        norms = self.quantizer._score_norms(self.compressed, None)
        fine_scores = _apply_norms(
            X_hat_cand @ q_rot,
            None if norms is None else norms[coarse_idx],
        ) + self.quantizer._query_offset(query)
        bias = self.quantizer._row_bias(self.compressed, None)
        if bias is not None:
            fine_scores = fine_scores + bias[coarse_idx]
        k_eff = min(k, len(coarse_idx))
        order = np.argsort(-fine_scores)[:k_eff]
        return coarse_idx[order], fine_scores[order]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def index_nbytes(self) -> int:
        """In-RAM bytes of the IVF structure (excluding the corpus)."""
        total = (
            self.cell_ids.nbytes
            + self.sorted_idx.nbytes
            + self.cell_offsets.nbytes
        )
        if self.hyperplanes is not None:
            total += self.hyperplanes.nbytes
        return int(total)

    @property
    def avg_cell_size(self) -> float:
        return self.n / self.n_cells if self.n_cells > 0 else 0.0

    def cell_size_stats(self) -> dict:
        """Summary statistics on cell occupancy (load balance)."""
        sizes = np.diff(self.cell_offsets)
        nonempty = sizes[sizes > 0]
        return {
            "n_cells": int(self.n_cells),
            "nonempty_cells": int((sizes > 0).sum()),
            "min": int(sizes.min()),
            "max": int(sizes.max()),
            "mean": float(sizes.mean()),
            "std": float(sizes.std()),
            "median": float(np.median(sizes)),
            "p95": float(np.percentile(sizes, 95)),
            "nonempty_min": int(nonempty.min()) if len(nonempty) else 0,
        }
