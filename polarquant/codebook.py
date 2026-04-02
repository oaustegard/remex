"""Lloyd-Max optimal scalar quantizer for post-rotation coordinate distribution."""

import numpy as np
from scipy.stats import norm
from typing import Tuple


def lloyd_max_codebook(
    d: int, bits: int, n_iter: int = 300
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build optimal Lloyd-Max codebook for N(0, 1/d) distributed coordinates.

    After random orthogonal rotation of unit vectors in R^d, each coordinate
    follows a distribution that concentrates to N(0, 1/d) as d grows.
    The Lloyd-Max quantizer minimizes MSE for this known distribution.

    Args:
        d: Vector dimension (determines coordinate variance = 1/d).
        bits: Quantization bit-width (produces 2^bits levels).
        n_iter: Lloyd-Max iteration count.

    Returns:
        boundaries: (2^bits - 1,) decision boundaries for np.searchsorted.
        centroids: (2^bits,) reconstruction values.
    """
    n_levels = 2**bits
    sigma = 1.0 / np.sqrt(d)
    rv = norm(0, sigma)

    centroids = np.linspace(-3 * sigma, 3 * sigma, n_levels)

    for _ in range(n_iter):
        bounds = np.concatenate(
            [[-np.inf], (centroids[:-1] + centroids[1:]) / 2, [np.inf]]
        )
        for j in range(n_levels):
            lo, hi = bounds[j], bounds[j + 1]
            prob = rv.cdf(hi) - rv.cdf(lo)
            if prob > 1e-15:
                centroids[j] = sigma**2 * (rv.pdf(lo) - rv.pdf(hi)) / prob

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return boundaries.astype(np.float32), centroids.astype(np.float32)


def theoretical_mse(d: int, bits: int) -> float:
    """Theoretical MSE upper bound from TurboQuant Theorem 1."""
    return np.sqrt(3 * np.pi) / 2 * 4 ** (-bits)


def theoretical_lower_bound(bits: int) -> float:
    """Information-theoretic lower bound on MSE (Theorem 3)."""
    return 4 ** (-bits)
