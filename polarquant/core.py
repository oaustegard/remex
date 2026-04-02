"""Core PolarQuant encoder/decoder."""

import numpy as np
from typing import Optional, Tuple
from polarquant.codebook import lloyd_max_codebook
from polarquant.rotation import haar_rotation


class CompressedVectors:
    """Container for quantized vector data with bit-packed index storage.

    Indices are stored in packed format to minimize memory. Unpacking
    happens lazily on first access and is cached for repeated use
    (e.g., multiple search queries against the same corpus).
    """

    __slots__ = ("_packed", "_indices_cache", "norms", "d", "bits", "n")

    def __init__(
        self,
        indices: np.ndarray,
        norms: np.ndarray,
        d: int,
        bits: int,
        *,
        packed: bool = False,
    ):
        """
        Args:
            indices: Either uint8 indices (n, d) or pre-packed bytes.
            norms: (n,) float32 vector norms.
            d: Original vector dimension.
            bits: Quantization bit width.
            packed: If True, indices are already in packed format.
        """
        if packed:
            self._packed = indices
        else:
            self._packed = pack_indices(indices, bits)
        self._indices_cache = None
        self.norms = norms
        self.d = d
        self.bits = bits
        self.n = norms.shape[0]

    @property
    def indices(self) -> np.ndarray:
        """Unpacked uint8 indices (n, d). Cached after first access."""
        if self._indices_cache is None:
            self._indices_cache = unpack_indices(self._packed, self.bits, self.d)
        return self._indices_cache

    @property
    def nbytes(self) -> int:
        """Actual memory footprint in bytes (packed indices + norms)."""
        return self._packed.nbytes + self.norms.nbytes

    @property
    def nbytes_unpacked(self) -> int:
        """Memory if indices were stored as uint8 (for comparison)."""
        return self.n * self.d + self.norms.nbytes

    @property
    def compression_ratio(self) -> float:
        """Compression vs float32 storage."""
        return (self.n * self.d * 4) / self.nbytes

    def save(self, path: str):
        """Save to compressed .npz file (packed format)."""
        np.savez_compressed(
            path,
            packed=self._packed,
            norms=self.norms,
            d=np.int32(self.d),
            bits=np.int32(self.bits),
        )

    @classmethod
    def load(cls, path: str) -> "CompressedVectors":
        """Load from .npz file."""
        data = np.load(path)
        if "packed" in data:
            return cls(
                data["packed"],
                data["norms"],
                int(data["d"]),
                int(data["bits"]),
                packed=True,
            )
        # Backward compat: old format stored unpacked "indices"
        return cls(
            data["indices"],
            data["norms"],
            int(data["d"]),
            int(data["bits"]),
        )


class PolarQuantizer:
    """
    Vector quantizer using random rotation + scalar quantization.

    Encodes vectors by:
    1. Normalizing to unit sphere (storing norms separately)
    2. Applying a random orthogonal rotation (makes coordinates ~N(0, 1/d))
    3. Scalar-quantizing each coordinate with a codebook

    Two modes:
    - **Data-oblivious** (default): Uses a theoretical Lloyd-Max codebook for
      N(0, 1/d). No training data needed. Based on TurboQuant (Zandieh et al.,
      ICLR 2026).
    - **Calibrated**: Call ``calibrate(sample)`` with a few hundred vectors to
      learn per-dimension codebooks via k-means. Improves recall on real
      embeddings where post-rotation coordinate distributions vary across
      dimensions.

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
        # Default: uniform codebook (1D arrays, broadcast across all dims)
        self.boundaries, self.centroids = lloyd_max_codebook(d, bits)

    def calibrate(self, X: np.ndarray, n_iter: int = 50) -> "PolarQuantizer":
        """
        Fit per-dimension codebooks from sample data.

        Rotates the sample, then runs 1D k-means independently on each
        coordinate to learn dimension-specific centroids. This captures
        per-dimension variance differences that the theoretical N(0, 1/d)
        codebook cannot.

        Recommended sample size: 500+ vectors for 4-bit, 100+ for 3-bit.
        Below ~200 vectors at 4-bit, the data-oblivious codebook may
        actually perform better due to noisy estimates.

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

    def _rotate(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize and rotate vectors. Returns (X_rot, norms)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[np.newaxis]
        if X.shape[1] != self.d:
            raise ValueError(f"Expected d={self.d}, got {X.shape[1]}")
        norms = np.linalg.norm(X, axis=1)
        X_unit = X / np.maximum(norms, 1e-8)[:, None]
        X_rot = X_unit @ self.R.T
        return X_rot, norms

    def encode(self, X: np.ndarray) -> CompressedVectors:
        """
        Quantize a batch of vectors.

        Args:
            X: (n, d) float array. Need not be unit-normalized.

        Returns:
            CompressedVectors with bit-packed indices and norms.
        """
        X_rot, norms = self._rotate(X)

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

    def decode(self, compressed: CompressedVectors) -> np.ndarray:
        """
        Reconstruct vectors from compressed representation.

        Args:
            compressed: CompressedVectors from encode().

        Returns:
            (n, d) float32 array of approximate vectors.
        """
        indices = compressed.indices  # lazy unpack + cache

        if self.calibrated:
            dim_idx = np.arange(self.d)[np.newaxis, :]
            X_hat_rot = self.centroids[dim_idx, indices]
        else:
            X_hat_rot = self.centroids[indices]

        X_hat_unit = X_hat_rot @ self.R
        return X_hat_unit * compressed.norms[:, None]

    def search(
        self,
        compressed: CompressedVectors,
        query: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors by approximate inner product.

        Operates in rotated space to avoid full dequantization.

        Args:
            compressed: Encoded corpus.
            query: (d,) query vector.
            k: Number of results.

        Returns:
            (indices, scores): top-k corpus indices and approximate inner products.
        """
        query = np.asarray(query, dtype=np.float32)
        q_rot = self.R @ query

        idx = compressed.indices  # lazy unpack + cache

        if self.calibrated:
            dim_idx = np.arange(self.d)[np.newaxis, :]
            X_hat_rot = self.centroids[dim_idx, idx]
        else:
            X_hat_rot = self.centroids[idx]

        scores = (X_hat_rot @ q_rot) * compressed.norms

        if k >= compressed.n:
            topk_idx = np.argsort(-scores)
        else:
            topk_idx = np.argpartition(-scores, k)[:k]
            topk_idx = topk_idx[np.argsort(-scores[topk_idx])]
        return topk_idx, scores[topk_idx]

    def mse(self, X: np.ndarray) -> float:
        """Compute mean per-vector reconstruction MSE (L2 squared)."""
        compressed = self.encode(X)
        X_hat = self.decode(compressed)
        return float(
            np.mean(np.sum((np.asarray(X, np.float32) - X_hat) ** 2, axis=1))
        )
