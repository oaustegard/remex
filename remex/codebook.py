"""Lloyd-Max optimal scalar quantizer for post-rotation coordinate distribution."""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple


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


def nested_codebooks(
    d: int, max_bits: int
) -> Dict[int, np.ndarray]:
    """
    Build nested centroid tables for Matryoshka-style bit precision.

    Encodes at max_bits precision. For each coarser bit level b < max_bits,
    derives centroids by probability-weighted grouping of the max_bits
    centroids. The top b bits of a max_bits index are a valid b-bit index
    into the corresponding centroid table.

    The Gaussian distribution is successively refinable, so the nesting
    penalty is small (typically <1.5% recall vs independently optimized
    codebooks at each level).

    Args:
        d: Vector dimension.
        max_bits: Maximum quantization bit-width.

    Returns:
        Dict mapping bit-width to centroid array:
        {max_bits: (2^max_bits,), max_bits-1: (2^(max_bits-1),), ..., 1: (2,)}
    """
    _, centroids_max = lloyd_max_codebook(d, max_bits)
    n_max = len(centroids_max)
    sigma = 1.0 / np.sqrt(d)
    rv = norm(0, sigma)

    # Probability mass for each max_bits bin
    bounds_max = (centroids_max[:-1] + centroids_max[1:]) / 2.0
    full_bounds = np.concatenate([[-np.inf], bounds_max, [np.inf]])
    probs = np.array(
        [rv.cdf(full_bounds[i + 1]) - rv.cdf(full_bounds[i]) for i in range(n_max)]
    )

    result = {max_bits: centroids_max}

    for target_bits in range(max_bits - 1, 0, -1):
        n_target = 2**target_bits
        group_size = n_max // n_target
        nested_centroids = np.empty(n_target, dtype=np.float32)

        for g in range(n_target):
            start = g * group_size
            end = start + group_size
            group_probs = probs[start:end]
            total_prob = group_probs.sum()
            if total_prob > 1e-15:
                nested_centroids[g] = np.average(
                    centroids_max[start:end], weights=group_probs
                )
            else:
                nested_centroids[g] = centroids_max[start:end].mean()

        result[target_bits] = nested_centroids

    return result


def theoretical_mse(d: int, bits: int) -> float:
    """Theoretical MSE upper bound from TurboQuant Theorem 1."""
    return np.sqrt(3 * np.pi) / 2 * 4 ** (-bits)


def theoretical_lower_bound(bits: int) -> float:
    """Information-theoretic lower bound on MSE (Theorem 3)."""
    return 4 ** (-bits)
