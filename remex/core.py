"""Core remex encoder/decoder with Matryoshka bit precision."""

import numpy as np
from typing import Optional, Tuple
from remex.codebook import lloyd_max_codebook, nested_codebooks
from remex.packing import pack, unpack, packed_nbytes
from remex.rotation import haar_rotation


class CompressedVectors:
    """Container for quantized vector data.

    Indices are stored as uint8 in memory for fast search/decode.
    Bit-packing is used for serialization (save/load) and for
    computing the true compressed size (nbytes).
    """

    __slots__ = ("indices", "norms", "d", "bits", "n", "_x_hat_rot")

    def __init__(self, indices: np.ndarray, norms: np.ndarray, d: int, bits: int):
        self.indices = indices  # (n, d) uint8 — unpacked for fast access
        self.norms = norms
        self.d = d
        self.bits = bits
        self.n = indices.shape[0]
        self._x_hat_rot = None  # cached dequantized rotated vectors (full precision)

    def subset(self, idx: np.ndarray) -> "CompressedVectors":
        """Return a CompressedVectors containing only the given row indices."""
        sub = CompressedVectors(
            self.indices[idx], self.norms[idx], self.d, self.bits
        )
        if self._x_hat_rot is not None:
            sub._x_hat_rot = self._x_hat_rot[idx]
        return sub

    @property
    def nbytes(self) -> int:
        """Packed memory footprint in bytes (honest compression)."""
        return packed_nbytes(self.n, self.d, self.bits) + self.norms.nbytes

    @property
    def nbytes_unpacked(self) -> int:
        """Unpacked memory footprint (what's actually in RAM)."""
        return self.indices.nbytes + self.norms.nbytes

    @property
    def compression_ratio(self) -> float:
        """Ratio vs float32 storage (using packed size)."""
        return (self.n * self.d * 4) / self.nbytes

    @property
    def resident_bytes(self) -> int:
        """Actual RAM footprint including any active caches."""
        total = self.indices.nbytes + self.norms.nbytes
        if self._x_hat_rot is not None:
            total += self._x_hat_rot.nbytes
        return total

    def drop_cache(self):
        """Free the dequantized float32 cache to reclaim memory."""
        self._x_hat_rot = None

    def save(self, path: str):
        """Save to compressed .npz file with bit-packed indices."""
        packed_idx = pack(self.indices.ravel(), self.bits)
        np.savez_compressed(
            path,
            packed_indices=packed_idx,
            norms=self.norms,
            d=np.int32(self.d),
            bits=np.int32(self.bits),
            n=np.int32(self.n),
        )

    @classmethod
    def load(cls, path: str) -> "CompressedVectors":
        """Load from .npz file, unpacking bit-packed indices."""
        data = np.load(path)
        d = int(data["d"])
        bits = int(data["bits"])
        n = int(data["n"])

        if "packed_indices" in data:
            flat = unpack(data["packed_indices"], bits, n * d)
            indices = flat.reshape(n, d)
        else:
            # Backward compat: old format stored unpacked indices
            indices = data["indices"]

        return cls(indices, data["norms"], d, bits)


class Quantizer:
    """
    Vector quantizer with Matryoshka bit precision.

    Encodes vectors by:
    1. Normalizing to unit sphere (storing norms separately)
    2. Applying a random orthogonal rotation (makes coordinates ~N(0, 1/d))
    3. Scalar-quantizing each coordinate with a Lloyd-Max codebook

    Data-oblivious: Uses a theoretical Lloyd-Max codebook for N(0, 1/d).
    No training data needed. Based on TurboQuant (Zandieh et al., ICLR 2026).

    Supports **nested bit precision**: encode once at full bit-width,
    then search at any lower precision by right-shifting indices.
    The top k bits of an n-bit code are a valid k-bit code, with
    centroid tables precomputed for each level.

    Args:
        d: Vector dimension.
        bits: Bits per coordinate (1-8). 3-4 is the sweet spot.
        seed: Random seed for rotation matrix.
    """

    def __init__(self, d: int, bits: int = 4, seed: int = 42):
        if bits < 1 or bits > 8:
            raise ValueError(f"bits must be 1-8, got {bits}")

        self.d = d
        self.bits = bits
        self.seed = seed

        self.R = haar_rotation(d, seed)
        self.boundaries, self.centroids = lloyd_max_codebook(d, bits)

        # Precompute nested centroid tables for all bit levels <= bits
        self._nested = nested_codebooks(d, bits)

    def encode(self, X: np.ndarray) -> CompressedVectors:
        """
        Quantize a batch of vectors.

        Args:
            X: (n, d) float array. Need not be unit-normalized.

        Returns:
            CompressedVectors container with indices and norms.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[np.newaxis]
        if X.shape[1] != self.d:
            raise ValueError(f"Expected d={self.d}, got {X.shape[1]}")

        norms = np.linalg.norm(X, axis=1)
        X_unit = X / np.maximum(norms, 1e-8)[:, None]
        X_rot = X_unit @ self.R.T

        indices = np.searchsorted(self.boundaries, X_rot).astype(np.uint8)

        return CompressedVectors(
            indices, norms.astype(np.float32), self.d, self.bits
        )

    def decode(
        self, compressed: CompressedVectors, precision: Optional[int] = None
    ) -> np.ndarray:
        """
        Reconstruct vectors from compressed representation.

        Args:
            compressed: CompressedVectors from encode().
            precision: Bit precision for reconstruction (1 to self.bits).
                       None = full precision. Only available in data-oblivious mode.

        Returns:
            (n, d) float32 array of approximate vectors.
        """
        centroids = self._resolve_centroids(compressed, precision)
        indices = self._resolve_indices(compressed, precision)
        X_hat_rot = centroids[indices]

        X_hat_unit = X_hat_rot @ self.R
        return X_hat_unit * compressed.norms[:, None]

    def search(
        self,
        compressed: CompressedVectors,
        query: np.ndarray,
        k: int = 10,
        precision: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors by approximate inner product.

        Operates in rotated space to avoid full dequantization.
        Caches the dequantized rotated representation for subsequent queries.

        Args:
            compressed: Encoded corpus.
            query: (d,) query vector.
            k: Number of results.
            precision: Bit precision for search (1 to self.bits).
                       Lower = faster/coarser, higher = more accurate.
                       None = full precision (self.bits).
                       Only available in data-oblivious mode.

        Returns:
            (indices, scores): top-k corpus indices and approximate scores.
        """
        query = np.asarray(query, dtype=np.float32)
        q_rot = self.R @ query

        X_hat_rot = self._get_x_hat_rot(compressed, precision)
        scores = (X_hat_rot @ q_rot) * compressed.norms

        if k >= compressed.n:
            topk_idx = np.argsort(-scores)
        else:
            topk_idx = np.argpartition(-scores, k)[:k]
            topk_idx = topk_idx[np.argsort(-scores[topk_idx])]
        return topk_idx, scores[topk_idx]

    def search_adc(
        self,
        compressed: CompressedVectors,
        query: np.ndarray,
        k: int = 10,
        precision: Optional[int] = None,
        chunk_size: int = 4096,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Memory-efficient search via asymmetric distance computation.

        Computes approximate inner products using a lookup table over
        the uint8 indices, without materializing an (n, d) float32 matrix.
        Peak temporary memory is chunk_size * d * 4 bytes (~6 MB default).

        Slower per-query than ``search()`` (no persistent cache), but
        uses dramatically less RAM. Ideal for the coarse stage of
        two-stage retrieval, or when memory is constrained.

        Args:
            compressed: Encoded corpus.
            query: (d,) query vector.
            k: Number of results.
            precision: Bit precision (1 to self.bits). None = full.
            chunk_size: Vectors per scoring chunk. Controls peak memory.

        Returns:
            (indices, scores): top-k corpus indices and approximate scores.
        """
        query = np.asarray(query, dtype=np.float32)
        q_rot = self.R @ query

        centroids = self._resolve_centroids(compressed, precision)
        indices = self._resolve_indices(compressed, precision)

        # Build ADC lookup table: table[j, i] = centroid_i * q_rot_j
        # centroids are (n_levels,), same for all dims
        table = np.outer(q_rot, centroids).astype(np.float32)
        # table shape: (d, n_levels)

        scores = self._adc_score_chunked(
            table, indices, compressed.norms, chunk_size
        )

        if k >= compressed.n:
            topk_idx = np.argsort(-scores)
        else:
            topk_idx = np.argpartition(-scores, k)[:k]
            topk_idx = topk_idx[np.argsort(-scores[topk_idx])]
        return topk_idx, scores[topk_idx]

    def search_twostage(
        self,
        compressed: CompressedVectors,
        query: np.ndarray,
        k: int = 10,
        candidates: int = 500,
        coarse_precision: Optional[int] = None,
        coarse_chunk_size: int = 4096,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Two-stage retrieval: memory-efficient coarse scan + precise rerank.

        Stage 1 (coarse): ADC lookup-table scan over the full corpus at
        reduced precision. No float32 cache — memory cost is only the
        uint8 indices (already stored) plus a tiny lookup table.

        Stage 2 (fine): Dequantize only the candidate vectors at full
        precision, then rerank by exact (quantized) inner product.

        Memory profile at 100k vectors, d=384:
          - Single-stage search():    154 MB  (cached n*d float32)
          - Two-stage search_twostage: ~39 MB  (uint8 indices + 6 MB temp)

        Args:
            compressed: Encoded corpus.
            query: (d,) query vector.
            k: Final number of results.
            candidates: Number of coarse candidates (stage 1).
            coarse_precision: Bit precision for coarse pass.
                              Default: max(1, self.bits - 2).
            coarse_chunk_size: Chunk size for ADC scoring.

        Returns:
            (indices, scores): top-k corpus indices and full-precision scores.
            Indices are into the original corpus (not the candidate set).
        """
        if coarse_precision is None:
            coarse_precision = max(1, self.bits - 2)

        query = np.asarray(query, dtype=np.float32)
        q_rot = self.R @ query
        coarse_k = min(candidates, compressed.n)

        # Stage 1: ADC coarse scan — no float32 cache needed
        coarse_centroids = self._resolve_centroids(compressed, coarse_precision)
        coarse_indices = self._resolve_indices(compressed, coarse_precision)

        coarse_table = np.outer(q_rot, coarse_centroids).astype(np.float32)

        coarse_scores = self._adc_score_chunked(
            coarse_table, coarse_indices, compressed.norms, coarse_chunk_size
        )

        if coarse_k >= compressed.n:
            coarse_idx = np.argsort(-coarse_scores)
        else:
            coarse_idx = np.argpartition(-coarse_scores, coarse_k)[:coarse_k]

        # Stage 2: full-precision rerank on small candidate set
        fine_centroids = self._resolve_centroids(compressed, None)
        fine_indices = compressed.indices[coarse_idx]

        X_hat_cand = fine_centroids[fine_indices]  # (candidates, d)

        fine_scores = (X_hat_cand @ q_rot) * compressed.norms[coarse_idx]
        rerank_order = np.argsort(-fine_scores)[:k]

        original_idx = coarse_idx[rerank_order]
        return original_idx, fine_scores[rerank_order]

    def mse(self, X: np.ndarray, precision: Optional[int] = None) -> float:
        """Compute mean per-vector reconstruction MSE (L2 squared)."""
        compressed = self.encode(X)
        X_hat = self.decode(compressed, precision=precision)
        return float(
            np.mean(np.sum((np.asarray(X, np.float32) - X_hat) ** 2, axis=1))
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adc_score_chunked(
        table: np.ndarray,
        indices: np.ndarray,
        norms: np.ndarray,
        chunk_size: int,
    ) -> np.ndarray:
        """Score vectors via ADC lookup table, processing in chunks.

        Args:
            table: (d, n_levels) float32 lookup table.
            indices: (n, d) uint8 quantization indices.
            norms: (n,) float32 vector norms.
            chunk_size: Rows per chunk (controls peak memory).

        Returns:
            (n,) float32 approximate inner-product scores.

        Memory: peak allocation is chunk_size * d * 4 bytes.
        At chunk_size=4096, d=384: ~6 MB temporary.
        """
        n = len(norms)
        d = table.shape[0]
        dim_idx = np.arange(d)
        scores = np.empty(n, dtype=np.float32)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_idx = indices[start:end]  # (chunk, d) uint8
            # Gather: table[j, chunk_idx[i, j]] → (chunk, d) float32
            # Then sum over d → (chunk,) inner-product contribution
            chunk_scores = table[dim_idx, chunk_idx].sum(axis=1)
            scores[start:end] = chunk_scores * norms[start:end]

        return scores

    def _get_x_hat_rot(
        self, compressed: CompressedVectors, precision: Optional[int] = None
    ) -> np.ndarray:
        """Get dequantized vectors in rotated space, with caching.

        Cache is only used for full-precision queries (precision=None).
        """
        if precision is None and compressed._x_hat_rot is not None:
            return compressed._x_hat_rot

        centroids = self._resolve_centroids(compressed, precision)
        indices = self._resolve_indices(compressed, precision)

        X_hat_rot = centroids[indices]

        # Cache only full-precision results
        if precision is None:
            compressed._x_hat_rot = X_hat_rot

        return X_hat_rot

    def _resolve_centroids(
        self, compressed: CompressedVectors, precision: Optional[int]
    ) -> np.ndarray:
        """Get centroid table for the requested precision level."""
        if precision is None:
            return self.centroids
        if precision < 1 or precision > self.bits:
            raise ValueError(
                f"precision must be 1-{self.bits}, got {precision}"
            )
        return self._nested[precision]

    def _resolve_indices(
        self, compressed: CompressedVectors, precision: Optional[int]
    ) -> np.ndarray:
        """Right-shift indices to the requested precision level."""
        if precision is None or precision == self.bits:
            return compressed.indices
        shift = self.bits - precision
        return compressed.indices >> shift
