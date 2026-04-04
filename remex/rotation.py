"""Random orthogonal rotation via Haar-distributed matrices."""

import numpy as np


def haar_rotation(d: int, seed: int = 42) -> np.ndarray:
    """
    Generate a Haar-distributed random orthogonal matrix.

    Uses QR decomposition of a Gaussian matrix, which produces a
    uniformly random orthogonal matrix (Haar measure on O(d)).

    The rotation is the core trick: it transforms any input distribution
    into one where coordinates are approximately i.i.d. N(0, 1/d),
    enabling optimal per-coordinate scalar quantization.

    Args:
        d: Matrix dimension.
        seed: Random seed for reproducibility.

    Returns:
        R: (d, d) orthogonal matrix, R @ R.T = I.
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d)).astype(np.float32)
    R, _ = np.linalg.qr(G)
    return R
