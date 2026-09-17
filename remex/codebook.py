"""Lloyd-Max optimal scalar quantizer for post-rotation coordinate distribution."""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional, Tuple


def coordinate_sigma(d: int, sigma: Optional[float] = None) -> float:
    """Resolve the coordinate standard deviation a codebook is built for.

    ``None`` means the unit-sphere default, ``1/sqrt(d)``: after a random
    orthogonal rotation of a *unit* vector in R^d each coordinate
    concentrates to N(0, 1/d). Scalar mode (``Quantizer(normalize=False)``)
    skips the normalization, so its coordinate scale is the caller's to
    declare and is passed in explicitly.
    """
    if sigma is None:
        return 1.0 / float(np.sqrt(d))
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be finite and positive, got {sigma!r}")
    return sigma


def lloyd_max_codebook(
    d: int, bits: int, n_iter: int = 300, sigma: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build optimal Lloyd-Max codebook for N(0, sigma) distributed coordinates.

    After random orthogonal rotation of unit vectors in R^d, each coordinate
    follows a distribution that concentrates to N(0, 1/d) as d grows.
    The Lloyd-Max quantizer minimizes MSE for this known distribution.

    Args:
        d: Vector dimension (determines the default coordinate variance = 1/d).
        bits: Quantization bit-width (produces 2^bits levels).
        n_iter: Lloyd-Max iteration count.
        sigma: Coordinate standard deviation to optimize for. ``None``
            (default) is ``1/sqrt(d)``, the unit-sphere value — pass an
            explicit sigma only when the coordinates were not unit-normalized
            first (see ``Quantizer(normalize=False)``).

    Returns:
        boundaries: (2^bits - 1,) decision boundaries for np.searchsorted.
        centroids: (2^bits,) reconstruction values.
    """
    n_levels = 2**bits
    sigma = coordinate_sigma(d, sigma)
    rv = norm(0, sigma)

    centroids = np.linspace(-3 * sigma, 3 * sigma, n_levels)

    for _ in range(n_iter):
        bounds = np.concatenate(
            [[-np.inf], (centroids[:-1] + centroids[1:]) / 2, [np.inf]]
        )
        # Vectorized Lloyd update.  `bounds` is materialised before any
        # centroid moves, so every cell's update reads the SAME boundaries --
        # a simultaneous (Jacobi) step, which is exactly what the per-level
        # loop did.  Evaluating cdf/pdf once over the whole boundary array
        # instead of four scalar scipy calls per level is bit-for-bit
        # identical and ~360x faster at d=768, bits=8 (14.7s -> 0.04s), where
        # the old form made 2^bits * n_iter * 4 = 307,200 scipy calls and
        # dominated `Quantizer.__init__` at roughly 50 of its 52 seconds.
        cdf = rv.cdf(bounds)
        pdf = rv.pdf(bounds)
        prob = np.diff(cdf)
        # E[X | cell] * P(cell) = sigma^2 * (phi(lo) - phi(hi)) for a Gaussian
        num = pdf[:-1] - pdf[1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            updated = sigma**2 * num / prob
        # Cells with negligible mass keep their previous centroid, as before.
        centroids = np.where(prob > 1e-15, updated, centroids)

    # Boundaries are midpoints of the float32 table, not of the float64
    # centroids. scipy's norm.cdf/pdf differ by an ulp across NumPy's SIMD
    # dispatch levels (x86-64-v4 vs v3, ARM, Accelerate); the float32 cast
    # absorbs that for the centroids but not for float64 midpoints, so the
    # old boundaries differed across machines at 4 and 8 bits (the 4-bit
    # middle boundary was 0.0 on one, 3.6e-17 on another). A float32 sum
    # widened to float64 is exact, so these depend only on the float32
    # centroids. Measured: experiments/rht-operator-native.
    c32 = centroids.astype(np.float32)
    c64 = c32.astype(np.float64)
    boundaries = ((c64[:-1] + c64[1:]) / 2.0).astype(np.float32)
    return boundaries, c32


def nested_codebooks(
    d: int, max_bits: int, sigma: Optional[float] = None
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
        sigma: Coordinate standard deviation, as in ``lloyd_max_codebook``.
            ``None`` (default) is the unit-sphere value ``1/sqrt(d)``.

    Returns:
        Dict mapping bit-width to centroid array:
        {max_bits: (2^max_bits,), max_bits-1: (2^(max_bits-1),), ..., 1: (2,)}
    """
    sigma = coordinate_sigma(d, sigma)
    _, centroids_max = lloyd_max_codebook(d, max_bits, sigma=sigma)
    n_max = len(centroids_max)
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
