"""Core polar-embed encoder/decoder with Matryoshka bit precision."""

import numpy as np
from typing import Optional, Tuple
from polar_embed.codebook import lloyd_max_codebook, nested_codebooks
from polar_embed.packing import pack, unpack, packed_nbytes
from polar_embed.rotation import haar_rotation


class CompressedVectors:
    """Container for quantized vector data.

    Indices are stored as uint8 in memory for fast search/decode.
    Bit-packing is used for serialization (save/load) and for
    computing the true compressed size (nbytes).
    """

    __slots__ = ("indices", "norms", "d", "bits", "n", "_x_hat_rot", "_x_hat_rot_cache")

    def __init__(self, indices: np.ndarray, norms: np.ndarray, d: int, bits: int):
        self.indices = indices  # (n, d) uint8 — unpacked for fast access
        self.norms = norms
        self.d = d
        self.bits = bits
        self.n = indices.shape[0]
        self._x_hat_rot = None  # cached dequantized rotated vectors (full precision)
        self._x_hat_rot_cache = {}  # precision → cached dequantized rotated vectors

    def subset(self, idx: np.ndarray) -> "CompressedVectors":
        """Return a CompressedVectors containing only the given row indices."""
        sub = CompressedVectors(
            self.indices[idx], self.norms[idx], self.d, self.bits
        )
        if self._x_hat_rot is not None:
            sub._x_hat_rot = self._x_hat_rot[idx]
        for prec, cached in self._x_hat_rot_cache.items():
            sub._x_hat_rot_cache[prec] = cached[idx]
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


class PolarQuantizer:
    """
    Vector quantizer with Matryoshka bit precision.

    Encodes vectors by:
    1. Normalizing to unit sphere (storing norms separately)
    2. Applying a random orthogonal rotation (makes coordinates ~N(0, 1/d))
    3. Scalar-quantizing each coordinate with a Lloyd-Max codebook

    Two modes:
    - **Data-oblivious** (default): Uses a theoretical Lloyd-Max codebook for
      N(0, 1/d). No training data needed. Based on TurboQuant (Zandieh et al.,
      ICLR 2026).
    - **Calibrated**: Call ``calibrate(sample)`` with a few hundred vectors to
      learn per-dimension codebooks via k-means. Improves recall on real
      embeddings where post-rotation coordinate distributions vary across
      dimensions.

    Supports **nested bit precision**: encode once at full bit-width,
    then search at any lower precision by right-shifting indices.
    The top k bits of an n-bit code are a valid k-bit code, with
    centroid tables precomputed for each level. Nesting penalty is
    typically <1.5% recall vs independently optimized codebooks.

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
        self.calibrated = False

        self.R = haar_rotation(d, seed)
        self.boundaries, self.centroids = lloyd_max_codebook(d, bits)

        # Precompute nested centroid tables for all bit levels <= bits
        self._nested = nested_codebooks(d, bits)

    def calibrate(self, X: np.ndarray, n_iter: int = 50) -> "PolarQuantizer":
        """
        Fit per-dimension codebooks from sample data.

        Rotates the sample, then runs 1D k-means independently on each
        coordinate to learn dimension-specific centroids. This captures
        per-dimension variance differences that the theoretical N(0, 1/d)
        codebook cannot.

        Calibration requires sufficient samples to outperform the
        data-oblivious codebook:
        - **4-bit**: ≥750 vectors (below ~400, calibration hurts recall)
        - **3-bit**: ≥100 vectors (benefits at smaller sample sizes)

        On real embeddings (all-MiniLM-L6-v2, d=384), full-corpus
        calibration improves R@10 by +2.9% at 4-bit and +3.8% at 3-bit.

        Note: Matryoshka nested precision is not available in calibrated mode,
        since per-dimension codebooks don't share the Gaussian nesting property.

        Args:
            X: (n, d) sample vectors (need not be the full corpus).
            n_iter: K-means iterations per dimension.

        Returns:
            self (for chaining: ``pq = PolarQuantizer(d, bits).calibrate(sample)``)
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[np.newaxis]
        if X.shape[1] != self.d:
            raise ValueError(f"Expected d={self.d}, got {X.shape[1]}")

        # Rotate sample
        norms = np.linalg.norm(X, axis=1)
        X_unit = X / np.maximum(norms, 1e-8)[:, None]
        X_rot = X_unit @ self.R.T

        n_levels = 2 ** self.bits
        centroids = np.zeros((self.d, n_levels), dtype=np.float32)
        boundaries = np.zeros((self.d, n_levels - 1), dtype=np.float32)

        for j in range(self.d):
            col = X_rot[:, j]
            c = np.percentile(
                col, np.linspace(0, 100, n_levels + 2)[1:-1]
            ).astype(np.float32)
            for _ in range(n_iter):
                b = (c[:-1] + c[1:]) / 2
                labels = np.searchsorted(b, col)
                new_c = np.empty_like(c)
                for lev in range(n_levels):
                    mask = labels == lev
                    new_c[lev] = col[mask].mean() if mask.sum() > 0 else c[lev]
                if np.allclose(c, new_c, atol=1e-7):
                    break
                c = new_c
            centroids[j] = c
            boundaries[j] = (c[:-1] + c[1:]) / 2

        self.centroids = centroids
        self.boundaries = boundaries
        self.calibrated = True
        return self

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

        if self.calibrated:
            indices = np.empty(X_rot.shape, dtype=np.uint8)
            for j in range(self.d):
                indices[:, j] = np.searchsorted(
                    self.boundaries[j], X_rot[:, j]
                )
        else:
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

        if self.calibrated and precision is None:
            dim_idx = np.arange(self.d)[np.newaxis, :]
            X_hat_rot = centroids[dim_idx, indices]
        else:
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

    def search_twostage(
        self,
        compressed: CompressedVectors,
        query: np.ndarray,
        k: int = 10,
        candidates: int = 500,
        coarse_precision: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Two-stage retrieval: coarse search then full-precision rerank.

        Stage 1: Search at coarse_precision for top `candidates`.
        Stage 2: Rerank candidates at full precision for top `k`.

        This leverages the Matryoshka bit nesting — the same encoded
        data is searched at two different precision levels.

        Both the coarse-precision and full-precision dequantized
        representations are cached on the CompressedVectors object,
        so repeated queries pay only the dot-product cost.

        Args:
            compressed: Encoded corpus.
            query: (d,) query vector.
            k: Final number of results.
            candidates: Number of coarse candidates (stage 1).
            coarse_precision: Bit precision for coarse pass.
                              Default: max(1, self.bits - 2).

        Returns:
            (indices, scores): top-k corpus indices and full-precision scores.
            Indices are into the original corpus (not the candidate set).
        """
        if coarse_precision is None:
            coarse_precision = max(1, self.bits - 2)

        query = np.asarray(query, dtype=np.float32)
        q_rot = self.R @ query
        coarse_k = min(candidates, compressed.n)

        # Stage 1: coarse pass at reduced precision (cached)
        X_coarse = self._get_x_hat_rot_at(compressed, coarse_precision)
        coarse_scores = (X_coarse @ q_rot) * compressed.norms
        if coarse_k >= compressed.n:
            coarse_idx = np.argsort(-coarse_scores)
        else:
            coarse_idx = np.argpartition(-coarse_scores, coarse_k)[:coarse_k]

        # Stage 2: rerank at full precision (cached)
        X_full = self._get_x_hat_rot(compressed)
        fine_scores = (X_full[coarse_idx] @ q_rot) * compressed.norms[coarse_idx]
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

        if self.calibrated and precision is None:
            dim_idx = np.arange(self.d)[np.newaxis, :]
            X_hat_rot = centroids[dim_idx, indices]
        else:
            X_hat_rot = centroids[indices]

        # Cache only full-precision results
        if precision is None:
            compressed._x_hat_rot = X_hat_rot

        return X_hat_rot

    def _get_x_hat_rot_at(
        self, compressed: CompressedVectors, precision: int
    ) -> np.ndarray:
        """Get dequantized vectors in rotated space at a specific precision, with caching."""
        if precision == self.bits:
            return self._get_x_hat_rot(compressed)

        if precision in compressed._x_hat_rot_cache:
            return compressed._x_hat_rot_cache[precision]

        centroids = self._resolve_centroids(compressed, precision)
        indices = self._resolve_indices(compressed, precision)
        X_hat_rot = centroids[indices]

        compressed._x_hat_rot_cache[precision] = X_hat_rot
        return X_hat_rot

    def _resolve_centroids(
        self, compressed: CompressedVectors, precision: Optional[int]
    ) -> np.ndarray:
        """Get centroid table for the requested precision level."""
        if precision is None:
            return self.centroids
        if self.calibrated:
            raise ValueError(
                "Matryoshka precision is not available in calibrated mode. "
                "Per-dimension codebooks don't support bit-level nesting."
            )
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
