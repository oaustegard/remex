"""Core polar-embed encoder/decoder."""

import numpy as np
from typing import Tuple
from polar_embed.codebook import lloyd_max_codebook
from polar_embed.rotation import haar_rotation


class CompressedVectors:
    """Container for quantized vector data."""

    __slots__ = ("indices", "norms", "d", "bits", "n")

    def __init__(self, indices: np.ndarray, norms: np.ndarray, d: int, bits: int):
        self.indices = indices
        self.norms = norms
        self.d = d
        self.bits = bits
        self.n = indices.shape[0]

    @property
    def nbytes(self) -> int:
        """Actual memory footprint in bytes."""
        return self.indices.nbytes + self.norms.nbytes

    @property
    def compression_ratio(self) -> float:
        """Ratio vs float32 storage."""
        return (self.n * self.d * 4) / self.nbytes

    def save(self, path: str):
        """Save to compressed .npz file."""
        np.savez_compressed(
            path,
            indices=self.indices,
            norms=self.norms,
            d=np.int32(self.d),
            bits=np.int32(self.bits),
        )

    @classmethod
    def load(cls, path: str) -> "CompressedVectors":
        """Load from .npz file."""
        data = np.load(path)
        return cls(data["indices"], data["norms"], int(data["d"]), int(data["bits"]))


class PolarQuantizer:
    """
    Data-oblivious vector quantizer using random rotation + Lloyd-Max.

    Encodes vectors by:
    1. Normalizing to unit sphere (storing norms separately)
    2. Applying a random orthogonal rotation (makes coordinates ~N(0, 1/d))
    3. Scalar-quantizing each coordinate with a Lloyd-Max codebook

    This is the MSE-optimal stage of TurboQuant (Zandieh et al., ICLR 2026).
    For embedding retrieval (nearest-neighbor search), this alone outperforms
    the full TurboQuant Prod variant because lower variance beats zero bias
    when only ranking matters.

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

        return CompressedVectors(indices, norms.astype(np.float32), self.d, self.bits)

    def decode(self, compressed: CompressedVectors) -> np.ndarray:
        """
        Reconstruct vectors from compressed representation.

        Args:
            compressed: CompressedVectors from encode().

        Returns:
            (n, d) float32 array of approximate vectors.
        """
        X_hat_rot = self.centroids[compressed.indices]
        X_hat_unit = X_hat_rot @ self.R  # R is orthogonal -> R^T inverts R.T
        return X_hat_unit * compressed.norms[:, None]

    def search(
        self, compressed: CompressedVectors, query: np.ndarray, k: int = 10
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

        X_hat_rot = self.centroids[compressed.indices]
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
        return float(np.mean(np.sum((np.asarray(X, np.float32) - X_hat) ** 2, axis=1)))
