"""Core remex encoder/decoder with Matryoshka bit precision."""

import numpy as np
from typing import Optional, Tuple, Iterable
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

    def save_arrow(self, path: str, seed: Optional[int] = None, **extra_metadata):
        """Save to Arrow IPC (Feather v2) format, packing indices for storage.

        Requires pyarrow (optional dependency).

        Args:
            path: Output file path.
            seed: Quantizer seed to store in schema metadata.
            **extra_metadata: Additional key-value pairs for schema metadata.
        """
        packed = PackedVectors.from_compressed(self)
        packed.save_arrow(path, seed=seed, **extra_metadata)

    @classmethod
    def load_arrow(cls, path: str) -> "CompressedVectors":
        """Load from Arrow IPC (Feather v2) format, unpacking to uint8 indices.

        Args:
            path: Arrow IPC file path.

        Returns:
            CompressedVectors with unpacked uint8 indices.
        """
        packed = PackedVectors.load_arrow(path)
        return packed.to_compressed()


class PackedVectors:
    """Memory-efficient packed storage for quantized vectors.

    Stores indices bit-packed in memory, unpacking rows on demand.
    Uses 2-4x less RAM than CompressedVectors for sub-byte bit widths.

    Use ``from_compressed()`` to convert from CompressedVectors,
    ``from_rows()`` to reconstruct from database rows, or
    ``load()`` / ``load_arrow()`` to read from disk.

    Search is supported via ``Quantizer.search_adc()`` and
    ``Quantizer.search_twostage()``.  Cached ``search()`` is not
    supported — use ``to_compressed()`` to convert back if needed.
    """

    __slots__ = ("_packed", "norms", "d", "bits", "n", "_row_bytes", "_row_aligned")

    def __init__(
        self,
        packed: np.ndarray,
        norms: np.ndarray,
        n: int,
        d: int,
        bits: int,
    ):
        """
        Args:
            packed: (n, row_bytes) uint8 array of bit-packed indices.
            norms: (n,) float32 array of vector norms.
            n: Number of vectors.
            d: Vector dimension.
            bits: Bits per coordinate.
        """
        self._packed = packed  # (n, row_bytes) uint8
        self.norms = norms
        self.n = n
        self.d = d
        self.bits = bits
        self._row_bytes = packed_nbytes(1, d, bits)
        self._row_aligned = (d * bits) % 8 == 0

    def unpack_rows(self, start: int, end: int) -> np.ndarray:
        """Decompress a contiguous row slice to uint8 indices.

        Args:
            start: First row index (inclusive).
            end: Last row index (exclusive).

        Returns:
            (end - start, d) uint8 array of indices.
        """
        n_rows = end - start
        if self._row_aligned:
            flat = self._packed[start:end].ravel()
            return unpack(flat, self.bits, n_rows * self.d).reshape(n_rows, self.d)
        else:
            result = np.empty((n_rows, self.d), dtype=np.uint8)
            for i in range(n_rows):
                result[i] = unpack(self._packed[start + i], self.bits, self.d)
            return result

    def unpack_at(self, idx: np.ndarray) -> np.ndarray:
        """Decompress arbitrary row indices to uint8 indices.

        Args:
            idx: Array of row indices to unpack.

        Returns:
            (len(idx), d) uint8 array of indices.
        """
        idx = np.asarray(idx)
        if idx.ndim == 0:
            idx = idx.reshape(1)
        rows = self._packed[idx]  # (len(idx), row_bytes)
        if self._row_aligned:
            flat = rows.ravel()
            return unpack(flat, self.bits, len(idx) * self.d).reshape(len(idx), self.d)
        else:
            result = np.empty((len(idx), self.d), dtype=np.uint8)
            for i in range(len(idx)):
                result[i] = unpack(rows[i], self.bits, self.d)
            return result

    @classmethod
    def from_compressed(cls, compressed: CompressedVectors) -> "PackedVectors":
        """Convert a CompressedVectors to packed in-memory format.

        Args:
            compressed: CompressedVectors with unpacked uint8 indices.

        Returns:
            PackedVectors with bit-packed indices.
        """
        n, d, bits = compressed.n, compressed.d, compressed.bits
        row_bytes = packed_nbytes(1, d, bits)
        row_aligned = (d * bits) % 8 == 0
        if row_aligned:
            packed_flat = pack(compressed.indices.ravel(), bits)
            packed = packed_flat.reshape(n, row_bytes)
        else:
            packed = np.empty((n, row_bytes), dtype=np.uint8)
            for i in range(n):
                packed[i] = pack(compressed.indices[i], bits)
        return cls(packed, compressed.norms.copy(), n, d, bits)

    @classmethod
    def from_rows(
        cls,
        rows: Iterable,
        norms: np.ndarray,
        d: int,
        bits: int,
    ) -> "PackedVectors":
        """Reconstruct from database packed byte rows.

        Args:
            rows: Iterable of bytes/bytearray/ndarray, one per vector.
            norms: (n,) float32 array of vector norms.
            d: Vector dimension.
            bits: Bits per coordinate.

        Returns:
            PackedVectors instance.
        """
        row_list = []
        for r in rows:
            if isinstance(r, (bytes, bytearray)):
                row_list.append(np.frombuffer(r, dtype=np.uint8))
            else:
                row_list.append(np.asarray(r, dtype=np.uint8))
        packed = np.stack(row_list)
        n = len(row_list)
        return cls(packed, np.asarray(norms, dtype=np.float32), n, d, bits)

    def at_precision(self, target_bits: int) -> "PackedVectors":
        """Derive a lower-bit representation via Matryoshka right-shift.

        Unpacks in chunks, right-shifts, and repacks at target_bits.

        Args:
            target_bits: Target bit precision (1 to self.bits).

        Returns:
            New PackedVectors at the target precision.
        """
        if target_bits < 1 or target_bits > self.bits:
            raise ValueError(
                f"target_bits must be 1-{self.bits}, got {target_bits}"
            )
        if target_bits == self.bits:
            return self

        shift = self.bits - target_bits
        row_bytes_target = packed_nbytes(1, self.d, target_bits)
        target_aligned = (self.d * target_bits) % 8 == 0
        packed_target = np.empty((self.n, row_bytes_target), dtype=np.uint8)

        chunk = 4096
        for start in range(0, self.n, chunk):
            end = min(start + chunk, self.n)
            indices = self.unpack_rows(start, end)
            shifted = (indices >> shift).astype(np.uint8)
            if target_aligned:
                packed_flat = pack(shifted.ravel(), target_bits)
                packed_target[start:end] = packed_flat.reshape(
                    end - start, row_bytes_target
                )
            else:
                for i in range(end - start):
                    packed_target[start + i] = pack(shifted[i], target_bits)

        return PackedVectors(
            packed_target, self.norms.copy(), self.n, self.d, target_bits
        )

    def to_compressed(self) -> CompressedVectors:
        """Convert to CompressedVectors by unpacking all indices."""
        indices = self.unpack_rows(0, self.n)
        return CompressedVectors(indices, self.norms.copy(), self.d, self.bits)

    def subset(self, idx: np.ndarray) -> "PackedVectors":
        """Return a PackedVectors containing only the given row indices."""
        idx = np.asarray(idx)
        return PackedVectors(
            self._packed[idx].copy(),
            self.norms[idx].copy(),
            len(idx),
            self.d,
            self.bits,
        )

    @property
    def nbytes(self) -> int:
        """Packed memory footprint in bytes."""
        return self._packed.nbytes + self.norms.nbytes

    @property
    def resident_bytes(self) -> int:
        """Actual RAM footprint (same as nbytes — no caches)."""
        return self._packed.nbytes + self.norms.nbytes

    @property
    def compression_ratio(self) -> float:
        """Ratio vs float32 storage."""
        return (self.n * self.d * 4) / self.nbytes

    def save(self, path: str):
        """Save to compressed .npz file."""
        np.savez_compressed(
            path,
            packed_indices=self._packed.ravel(),
            norms=self.norms,
            d=np.int32(self.d),
            bits=np.int32(self.bits),
            n=np.int32(self.n),
        )

    @classmethod
    def load(cls, path: str) -> "PackedVectors":
        """Load from .npz file, keeping indices packed."""
        data = np.load(path)
        d = int(data["d"])
        bits = int(data["bits"])
        n = int(data["n"])
        row_bytes = packed_nbytes(1, d, bits)
        packed_flat = data["packed_indices"]
        packed = packed_flat.reshape(n, row_bytes)
        return cls(packed, data["norms"], n, d, bits)

    def save_arrow(self, path: str, seed: Optional[int] = None, **extra_metadata):
        """Save to Arrow IPC (Feather v2) format.

        Stores packed indices as FixedSizeBinary and norms as Float32,
        with quantizer parameters in schema-level metadata.

        Requires pyarrow (optional dependency).

        Args:
            path: Output file path.
            seed: Quantizer seed to store in metadata.
            **extra_metadata: Additional key-value pairs for schema metadata.
        """
        try:
            import pyarrow as pa
            import pyarrow.feather as feather
        except ImportError:
            raise ImportError(
                "pyarrow is required for Arrow IPC format: "
                "pip install pyarrow"
            )

        row_bytes = self._row_bytes
        metadata = {
            b"d": str(self.d).encode(),
            b"bits": str(self.bits).encode(),
            b"n": str(self.n).encode(),
        }
        if seed is not None:
            metadata[b"seed"] = str(seed).encode()
        for k, v in extra_metadata.items():
            key = k.encode() if isinstance(k, str) else k
            metadata[key] = str(v).encode()

        schema = pa.schema(
            [
                pa.field("norms", pa.float32()),
                pa.field("packed_indices", pa.binary(row_bytes)),
            ],
            metadata=metadata,
        )

        # Build arrays from flat buffers for efficiency
        norms_arr = pa.array(self.norms.tolist(), type=pa.float32())
        packed_buf = pa.py_buffer(self._packed.tobytes())
        packed_arr = pa.FixedSizeBinaryArray.from_buffers(
            pa.binary(row_bytes), self.n, [None, packed_buf]
        )

        table = pa.table(
            {"norms": norms_arr, "packed_indices": packed_arr}, schema=schema
        )
        feather.write_feather(table, path)

    @classmethod
    def load_arrow(cls, path: str, memory_map: bool = False) -> "PackedVectors":
        """Load from Arrow IPC (Feather v2) format.

        Args:
            path: Arrow IPC file path.
            memory_map: If True, memory-map the file for zero-copy access.

        Returns:
            PackedVectors with packed indices.
        """
        try:
            import pyarrow.feather as feather
        except ImportError:
            raise ImportError(
                "pyarrow is required for Arrow IPC format: "
                "pip install pyarrow"
            )

        table = feather.read_table(path, memory_map=memory_map)
        metadata = table.schema.metadata
        d = int(metadata[b"d"])
        bits = int(metadata[b"bits"])
        n = int(metadata[b"n"])
        row_bytes = packed_nbytes(1, d, bits)

        norms = table.column("norms").to_numpy().astype(np.float32)

        # Extract packed data from contiguous Arrow buffer
        packed_col = table.column("packed_indices")
        chunk = packed_col.chunk(0)
        buffers = chunk.buffers()
        data_buf = buffers[1]
        packed_flat = np.frombuffer(data_buf, dtype=np.uint8)[: n * row_bytes]
        packed = packed_flat.reshape(n, row_bytes).copy()

        return cls(packed, norms, n, d, bits)


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
            raise ValueError(f"bits must be 1-4 or 8, got {bits}")
        if bits in (5, 6, 7):
            raise ValueError(
                f"bits={bits} is not supported. Use 1-4 or 8 bits. "
                f"5-7 bit widths offer negligible benefit over 4-bit or 8-bit."
            )

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
            compressed: Encoded corpus (CompressedVectors only).
            query: (d,) query vector.
            k: Number of results.
            precision: Bit precision for search (1 to self.bits).
                       Lower = faster/coarser, higher = more accurate.
                       None = full precision (self.bits).
                       Only available in data-oblivious mode.

        Returns:
            (indices, scores): top-k corpus indices and approximate scores.

        Raises:
            TypeError: If passed a PackedVectors (use search_adc or
                search_twostage instead, or convert with to_compressed()).
        """
        if isinstance(compressed, PackedVectors):
            raise TypeError(
                "PackedVectors does not support cached search(). "
                "Use search_adc() or search_twostage(), or convert "
                "with packed.to_compressed() first."
            )
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
        compressed,
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

        Accepts both CompressedVectors and PackedVectors. When given
        PackedVectors, indices are unpacked on demand per chunk.

        Args:
            compressed: Encoded corpus (CompressedVectors or PackedVectors).
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
        table = np.outer(q_rot, centroids).astype(np.float32)

        if isinstance(compressed, PackedVectors):
            shift = 0 if (precision is None or precision == self.bits) else (self.bits - precision)
            scores = self._adc_score_packed(
                table, compressed, shift, chunk_size
            )
        else:
            indices = self._resolve_indices(compressed, precision)
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
        compressed,
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

        Accepts both CompressedVectors and PackedVectors. When given
        PackedVectors, indices are unpacked on demand per chunk (coarse)
        and only for the candidate rows (fine).

        Memory profile at 100k vectors, d=384:
          - Single-stage search():    154 MB  (cached n*d float32)
          - Two-stage search_twostage: ~39 MB  (uint8 indices + 6 MB temp)

        Args:
            compressed: Encoded corpus (CompressedVectors or PackedVectors).
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
        is_packed = isinstance(compressed, PackedVectors)

        # Stage 1: ADC coarse scan — no float32 cache needed
        coarse_centroids = self._resolve_centroids(compressed, coarse_precision)
        coarse_table = np.outer(q_rot, coarse_centroids).astype(np.float32)

        if is_packed:
            coarse_shift = self.bits - coarse_precision
            coarse_scores = self._adc_score_packed(
                coarse_table, compressed, coarse_shift, coarse_chunk_size
            )
        else:
            coarse_indices = self._resolve_indices(compressed, coarse_precision)
            coarse_scores = self._adc_score_chunked(
                coarse_table, coarse_indices, compressed.norms, coarse_chunk_size
            )

        if coarse_k >= compressed.n:
            coarse_idx = np.argsort(-coarse_scores)
        else:
            coarse_idx = np.argpartition(-coarse_scores, coarse_k)[:coarse_k]

        # Stage 2: full-precision rerank on small candidate set
        fine_centroids = self._resolve_centroids(compressed, None)
        if is_packed:
            fine_indices = compressed.unpack_at(coarse_idx)
        else:
            fine_indices = compressed.indices[coarse_idx]

        X_hat_cand = fine_centroids[fine_indices]  # (candidates, d)

        fine_scores = (X_hat_cand @ q_rot) * compressed.norms[coarse_idx]
        rerank_order = np.argsort(-fine_scores)[:k]

        original_idx = coarse_idx[rerank_order]
        return original_idx, fine_scores[rerank_order]

    def search_batch(
        self,
        compressed: CompressedVectors,
        queries: np.ndarray,
        k: int = 10,
        precision: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for a batch of queries.

        Uses matrix multiplication instead of per-query matvec for
        significantly better throughput via BLAS-level parallelism.

        Args:
            compressed: Encoded corpus (CompressedVectors only).
            queries: (n_queries, d) query matrix.
            k: Number of results per query.
            precision: Bit precision for search (1 to self.bits). None = full.

        Returns:
            (indices, scores): both (n_queries, k) arrays.
                indices[i] = top-k corpus indices for query i.
                scores[i] = corresponding approximate scores, descending.

        Raises:
            TypeError: If passed a PackedVectors.
        """
        if isinstance(compressed, PackedVectors):
            raise TypeError(
                "PackedVectors does not support cached search_batch(). "
                "Use search_adc() or search_twostage(), or convert "
                "with packed.to_compressed() first."
            )
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[np.newaxis]

        n_queries = queries.shape[0]
        Q_rot = queries @ self.R.T  # (n_queries, d)

        X_hat_rot = self._get_x_hat_rot(compressed, precision)
        # (n_queries, n) = (n_queries, d) @ (d, n)
        all_scores = (Q_rot @ X_hat_rot.T) * compressed.norms[np.newaxis, :]

        all_indices = np.empty((n_queries, min(k, compressed.n)), dtype=np.intp)
        all_topk_scores = np.empty((n_queries, min(k, compressed.n)), dtype=np.float32)

        for i in range(n_queries):
            scores_i = all_scores[i]
            if k >= compressed.n:
                topk_idx = np.argsort(-scores_i)
            else:
                topk_idx = np.argpartition(-scores_i, k)[:k]
                topk_idx = topk_idx[np.argsort(-scores_i[topk_idx])]
            all_indices[i] = topk_idx
            all_topk_scores[i] = scores_i[topk_idx]

        return all_indices, all_topk_scores

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

    @staticmethod
    def _adc_score_packed(
        table: np.ndarray,
        packed: "PackedVectors",
        shift: int,
        chunk_size: int,
    ) -> np.ndarray:
        """Score packed vectors via ADC, unpacking chunks on demand.

        Args:
            table: (d, n_levels) float32 lookup table.
            packed: PackedVectors with bit-packed indices.
            shift: Right-shift to apply for precision reduction (0 = full).
            chunk_size: Rows per chunk (controls peak memory).

        Returns:
            (n,) float32 approximate inner-product scores.
        """
        n = packed.n
        d = table.shape[0]
        dim_idx = np.arange(d)
        scores = np.empty(n, dtype=np.float32)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_idx = packed.unpack_rows(start, end)  # (chunk, d) uint8
            if shift > 0:
                chunk_idx = chunk_idx >> shift
            chunk_scores = table[dim_idx, chunk_idx].sum(axis=1)
            scores[start:end] = chunk_scores * packed.norms[start:end]

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
