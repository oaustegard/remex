"""Random orthogonal rotation via Haar-distributed matrices.

Uses an explicit Householder QR with a fixed reflector convention so the
output is bit-reproducible across BLAS builds and matches the Mojo port's
encode path byte-for-byte (issue #40). The previous implementation
delegated to ``np.linalg.qr``, which calls LAPACK ``dgeqrf``; LAPACK QR
is not bit-deterministic across MKL/OpenBLAS builds or threading modes,
which made `--seed`-based reproducibility impossible end-to-end.
"""

import math

import numpy as np

#: What an *absent* rotation record on disk means.
#:
#: A frozen historical constant, deliberately NOT tied to ``Quantizer``'s
#: live ``rotation`` default. Every index written before this field existed
#: was built with Haar, and must keep decoding as Haar no matter what the
#: default becomes later. Binding this to the default would mean flipping
#: the default silently re-decodes every stored index under a rotation that
#: did not encode it — and the codes are 50%-different, not slightly off,
#: so the failure is total and silent rather than noisy.
LEGACY_ROTATION = "haar"

#: Rotation name to its on-disk code, for the byte-oriented ``.pq`` and
#: ``.params`` headers. Haar is 0 because the reserved header bytes this
#: lives in were already zero-filled in every file written before the field
#: existed — so an old file decodes to ``LEGACY_ROTATION`` for free, with no
#: "is the field present?" branch.
#:
#: ``"none"`` is the identity — no rotation at all. It exists for
#: ``Quantizer(normalize=False)``, where the caller has already conditioned
#: the coordinate distribution and wants codes that are an exact,
#: matmul-free function of the input (see ``identity_rotation``).
ROTATION_CODES = {"haar": 0, "rht": 1, "none": 2}
ROTATION_BY_CODE = {code: name for name, code in ROTATION_CODES.items()}


def validate_rotation(kind: str) -> str:
    """Return ``kind`` if it names a rotation this library can build."""
    if kind not in ROTATION_CODES:
        raise ValueError(
            f"unknown rotation {kind!r}; expected one of "
            f"{sorted(ROTATION_CODES)}."
        )
    return kind


def rotation_from_code(code: int) -> str:
    """Map an on-disk rotation byte back to its name."""
    if code not in ROTATION_BY_CODE:
        raise ValueError(
            f"unknown rotation code {code} in header; this file was written "
            f"by a newer remex. Known codes: {sorted(ROTATION_BY_CODE)}."
        )
    return ROTATION_BY_CODE[code]


def _householder_qr(A: np.ndarray) -> np.ndarray:
    """In-place Householder QR; returns Q. After the call A holds R.

    Reflector convention (must match ``remex/mojo/src/rotation.mojo``):
      - ``alpha = -sign(A[k,k]) * ||A[k:, k]||`` with ``sign(0) = +1``
      - ``v = A[k:, k] - alpha * e_1``, normalized
      - Apply ``H = I - 2 v v^T`` to ``A[k:, k:]`` from the left and to
        ``Q[:, k:]`` from the right
    """
    n = A.shape[0]
    Q = np.eye(n, dtype=np.float64)
    for k in range(n - 1):
        col = A[k:, k]
        col_norm = float(np.sqrt(np.dot(col, col)))
        if col_norm == 0.0:
            continue
        sign_akk = -1.0 if A[k, k] < 0.0 else 1.0
        alpha = -sign_akk * col_norm

        v = col.copy()
        v[0] -= alpha
        v_norm = float(np.sqrt(np.dot(v, v)))
        if v_norm == 0.0:
            continue
        v /= v_norm

        # H = I - 2 v v^T applied to A[k:, k:] from the left
        # A[k:, k:] -= 2 * outer(v, v.T @ A[k:, k:])
        sub = A[k:, k:]
        sub -= 2.0 * np.outer(v, v @ sub)

        # Same H applied to Q[:, k:] from the right
        # Q[:, k:] -= 2 * outer(Q[:, k:] @ v, v)
        Qsub = Q[:, k:]
        Qsub -= 2.0 * np.outer(Qsub @ v, v)
    return Q


def haar_rotation(d: int, seed: int = 42) -> np.ndarray:
    """
    Generate a Haar-distributed random orthogonal matrix.

    Pipeline (matches the Mojo port for byte-identical encode parity):
      1. Sample G ~ N(0, 1) of shape (d, d) in float64 via NumPy's
         default RNG (PCG64 + SeedSequence + Ziggurat).
      2. Run explicit Householder QR with the fixed reflector convention
         documented in ``_householder_qr``.
      3. Apply Mezzadri sign correction so Q is uniformly distributed
         on O(d) (the Haar measure).
      4. Cast Q to float32.

    Args:
        d: Matrix dimension.
        seed: Random seed for reproducibility.

    Returns:
        Q: (d, d) float32 orthogonal matrix, Q @ Q.T ~ I.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))  # float64 by default
    Q = _householder_qr(A)

    # Mezzadri sign correction: ensure diag(R) > 0 so Q is Haar-distributed.
    diag = np.diagonal(A).copy()
    sign_flip = diag < 0.0
    if sign_flip.any():
        Q[:, sign_flip] = -Q[:, sign_flip]

    return Q.astype(np.float32)


def identity_rotation(d: int, seed: int = 42) -> np.ndarray:
    """The (d, d) identity — ``rotation="none"``, i.e. no rotation at all.

    Same signature as the real rotations so it drops into
    ``Quantizer.ROTATIONS`` unchanged; ``seed`` is accepted and ignored,
    because there is nothing random to seed.

    Quantizing without a rotation gives up the property the rotation exists
    for — that coordinates become approximately i.i.d. Gaussian, which is
    what makes one data-oblivious Lloyd-Max codebook right for all of them.
    It buys back two things that matter for the scalar/hash-key mode
    (``Quantizer(normalize=False)``):

    - **Exactness.** ``Quantizer`` skips the matmul entirely when R is the
      identity, so a code is a pure ``searchsorted`` of the input value. No
      floating-point summation order enters the encoding, so identical
      inputs give identical codes across BLAS builds and thread counts, not
      just within one process — which is the contract a hash key needs.
    - **Interpretability.** Coordinate *j* of the code depends only on
      coordinate *j* of the input, so a per-coordinate range argument
      (bounded values, an ``arcsinh`` pre-transform) carries straight
      through to the codes.

    Use it when the caller has already conditioned each coordinate to a
    comparable scale. Prefer ``"haar"`` or ``"rht"`` otherwise.
    """
    return np.eye(d, dtype=np.float32)


def _fwht_inplace(y: np.ndarray) -> None:
    """Normalized fast Walsh-Hadamard transform along the last axis, in place.

    ``y`` must be (..., B) with B a power of two.  Normalized by 1/sqrt(B) by
    the caller, which makes the transform orthogonal *and* symmetric — it is
    its own inverse.
    """
    B = y.shape[-1]
    h = 1
    while h < B:
        y2 = y.reshape(-1, B // (2 * h), 2, h)
        a = y2[:, :, 0, :].copy()
        b = y2[:, :, 1, :]
        y2[:, :, 0, :] = a + b
        y2[:, :, 1, :] = a - b
        h *= 2


def _largest_pow2_divisor(d: int) -> int:
    b = 1
    while d % (b * 2) == 0:
        b *= 2
    return b


def rht_rotation(d: int, seed: int = 42) -> np.ndarray:
    """Randomized Hadamard rotation, materialized as a dense (d, d) matrix.

    Same contract as ``haar_rotation``: a deterministic-from-seed float32
    orthogonal matrix.  It is *not* Haar-distributed — it is a randomized
    Hadamard transform, which is the standard incoherence-processing rotation
    in the QuIP#/HIGGS lineage and is what the coordinates-become-Gaussian
    argument actually needs.  Measured indistinguishable from Haar on
    retrieval recall (-0.0001 +/- 0.0013 pooled over 3 corpora x 6 bit widths
    x 5 seeds; oaustegard/experiments#11).

    Why it exists: ``haar_rotation`` runs an explicit Householder QR, which is
    O(d^3) and measured at O(d^3.08) here — 1.8 s at d=768, 11.4 s at d=1536,
    150 s at d=3072.  remex is embedding-agnostic and d is a free parameter,
    so that is a real cost at mainstream embedding sizes.  Building the RHT
    costs O(d^2 log d).

    Construction, for ANY d rather than powers of two only: rounds of
    (permute -> sign flip -> block-diagonal FWHT) with block size the largest
    power of two dividing d, enough rounds to mix every coordinate.  Padding d
    up to a power of two would change the codec's dimension, so it is not an
    option here.  Materialized by applying the transform to the identity, one
    batched pass.

    The Mojo port implements the same construction, off the same NumPy PCG64
    stream (``mojo/src/rotation.mojo::rht_rotation``), so ``polarquant
    --seed S --rotation rht`` rebuilds this matrix byte-for-byte and encodes
    identically.  ``--params`` works too, and reads R straight out of the
    file.

    Args:
        d: Matrix dimension.
        seed: Random seed. Same seed gives the same matrix.

    Returns:
        Q: (d, d) float32 orthogonal matrix, Q @ Q.T ~ I.
    """
    B = _largest_pow2_divisor(d)
    if B < 2:
        raise ValueError(
            f"d={d} is odd; the randomized Hadamard construction needs an "
            f"even dimension. Use rotation='haar'."
        )
    rng = np.random.default_rng(seed)
    rounds = 1 if B == d else max(2, math.ceil(math.log(d) / math.log(B)))

    # Apply the transform to the identity, one batched pass over all d rows.
    Y = np.eye(d, dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(B))
    for _ in range(rounds):
        perm = rng.permutation(d)
        sign = rng.choice(np.array([-1.0, 1.0], np.float32), size=d)
        Y = Y[:, perm] * sign
        Y = np.ascontiguousarray(Y.reshape(d, d // B, B))
        _fwht_inplace(Y)
        Y = Y.reshape(d, d) * scale
    return Y


# ── Operator form of the randomized Hadamard rotation ────────────────────
#
# ``rht_rotation`` above materializes the transform so that every consumer
# sees an ordinary matrix. ``RHTOperator`` applies the same transform
# directly: O(d log d) per row instead of O(d^2), a few kilobytes of state
# instead of d^2 floats, and output that is bit-identical on every machine
# (gathers, multiplies and pairwise add/sub only, in a fixed order). The
# dense matrix and the operator agree to float32 rounding (~3e-6); codes
# computed through them agree on all but ~1e-6 of coordinates.
#
# Measured against dense BLAS on x86, ARM and Apple Silicon:
# oaustegard/experiments rht-operator-native/RESULTS.md.

import os as _os
import threading as _threading
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor

#: Rows are split across threads only when a call carries at least this many
#: input floats; below it, waking a thread costs more than it saves.
#: Override with ``REMEX_PARALLEL_MIN``.
PARALLEL_MIN_FLOATS = int(_os.environ.get("REMEX_PARALLEL_MIN", 1 << 15))

_num_threads = None
_pool = None
_pool_pid = None
_pool_lock = _threading.Lock()


def _default_threads() -> int:
    env = _os.environ.get("REMEX_NUM_THREADS")
    if env:
        return max(1, int(env))
    try:
        return max(1, len(_os.sched_getaffinity(0)))
    except AttributeError:  # macOS, Windows
        return max(1, _os.cpu_count() or 1)


def set_num_threads(n) -> None:
    """Threads used by the RHT operator. ``None`` restores the default
    (``REMEX_NUM_THREADS``, else the CPUs this process may run on)."""
    global _num_threads
    _num_threads = None if n is None else max(1, int(n))


def get_num_threads() -> int:
    return _num_threads if _num_threads is not None else _default_threads()


def _executor(workers: int):
    """A per-process pool. A forked child gets a fresh one: the parent's
    worker threads do not exist there, and submitting to them would hang."""
    global _pool, _pool_pid
    pid = _os.getpid()
    with _pool_lock:
        if _pool is None or _pool_pid != pid or _pool._max_workers != workers:
            _pool = _ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="remex-rht")
            _pool_pid = pid
        return _pool


def rht_plan(d: int, seed: int = 42):
    """The seed-derived structure of ``rht_rotation(d, seed)``.

    Draws from the same PCG64 stream in the same order, so the operator and
    the materialized matrix are the same transform.

    Returns ``(B, rounds, perms)`` with ``perms`` an int32 ``(rounds, d)``
    array and a list of per-round sign vectors (float64, +/-1).
    """
    B = _largest_pow2_divisor(d)
    if B < 2:
        raise ValueError(
            f"d={d} is odd; the randomized Hadamard construction needs an "
            f"even dimension. Use rotation='haar'."
        )
    rng = np.random.default_rng(seed)
    rounds = 1 if B == d else max(2, math.ceil(math.log(d) / math.log(B)))
    perms, signs = [], []
    for _ in range(rounds):
        perms.append(rng.permutation(d))
        signs.append(rng.choice(np.array([-1.0, 1.0], np.float32), size=d))
    return B, rounds, np.ascontiguousarray(np.stack(perms), dtype=np.int32), signs


def _numpy_fwht_rows(Y: np.ndarray, B: int) -> np.ndarray:
    """Unnormalized block FWHT of each row, pairing exactly as the C kernel."""
    n, d = Y.shape
    V = Y.reshape(n, d // B, B)
    h = 1
    while h < B:
        V2 = V.reshape(n, d // B, B // (2 * h), 2, h)
        a = V2[..., 0, :].copy()
        b = V2[..., 1, :].copy()
        V2[..., 0, :] = a + b
        V2[..., 1, :] = a - b
        h *= 2
    return V.reshape(n, d)


class RHTOperator:
    """``rht_rotation(d, seed)`` applied without materializing it.

    ``rotate_rows(X) == X @ R.T`` and ``unrotate_rows(X) == X @ R`` up to
    float32 rounding, for float32 or float64 input (the result keeps the
    input's dtype). Uses the compiled kernel from :mod:`remex._native` when
    it is available and a NumPy implementation with identical bits
    otherwise.
    """

    def __init__(self, d: int, seed: int = 42):
        self.d = int(d)
        self.seed = seed
        B, rounds, perms, signs = rht_plan(self.d, seed)
        self.B, self.rounds, self.perms = B, rounds, perms
        inv_sqrt = 1.0 / math.sqrt(B)
        self._ss = {
            np.dtype(np.float32): np.ascontiguousarray(
                np.stack([s * np.float32(inv_sqrt) for s in signs]), dtype=np.float32),
            np.dtype(np.float64): np.ascontiguousarray(
                np.stack([s.astype(np.float64) * inv_sqrt for s in signs]), dtype=np.float64),
        }

    @property
    def nbytes(self) -> int:
        return self.perms.nbytes + sum(a.nbytes for a in self._ss.values())

    # -- public -----------------------------------------------------------
    def rotate_rows(self, X: np.ndarray) -> np.ndarray:
        """``X @ R.T``."""
        return self._apply(X, 1)

    def unrotate_rows(self, X: np.ndarray) -> np.ndarray:
        """``X @ R``."""
        return self._apply(X, 0)

    def rotate_query(self, q: np.ndarray) -> np.ndarray:
        """``R @ q`` for a single vector."""
        return self._apply(q, 1)

    def materialize(self, dtype=np.float32) -> np.ndarray:
        """The (d, d) matrix this operator applies (rows of ``I @ R``)."""
        return self.unrotate_rows(np.eye(self.d, dtype=dtype))

    # -- internals --------------------------------------------------------
    def _apply(self, X, mode: int) -> np.ndarray:
        X = np.asarray(X)
        if X.dtype not in (np.float32, np.float64):
            X = X.astype(np.float32)
        one = X.ndim == 1
        X2 = np.ascontiguousarray(X.reshape(1, -1) if one else X)
        if X2.ndim != 2 or X2.shape[1] != self.d:
            raise ValueError(f"Expected d={self.d}, got shape {X.shape}")
        k = _native_kernel()
        out = self._apply_native(k, X2, mode) if k is not None else self._apply_numpy(X2, mode)
        return out[0] if one else out

    def _apply_numpy(self, X: np.ndarray, mode: int) -> np.ndarray:
        ss = self._ss[X.dtype]
        Y = X
        order = range(self.rounds) if mode == 0 else reversed(range(self.rounds))
        for r in order:
            perm, s = self.perms[r], ss[r]
            if mode == 0:
                Y = _numpy_fwht_rows(Y[:, perm] * s, self.B)
            else:
                Y = _numpy_fwht_rows(np.array(Y, copy=True), self.B)
                Z = np.empty_like(Y)
                Z[:, perm] = Y * s
                Y = Z
        return Y

    def _apply_native(self, k, X: np.ndarray, mode: int) -> np.ndarray:
        fn = k.f32 if X.dtype == np.float32 else k.f64
        ss = self._ss[X.dtype]
        n, d = X.shape
        out = np.empty_like(X)
        args = (d, self.B, self.rounds, self.perms.ctypes.data, ss.ctypes.data, mode)
        xp, op = X.ctypes.data, out.ctypes.data
        threads = get_num_threads()
        if threads <= 1 or n < 2 or n * d < PARALLEL_MIN_FLOATS:
            buf = np.empty(d, X.dtype)
            fn(xp, op, 0, n, *args, buf.ctypes.data)
            return out
        threads = min(threads, n)
        step = -(-n // threads)

        def work(lo):
            buf = np.empty(d, X.dtype)
            fn(xp, op, lo, min(n, lo + step), *args, buf.ctypes.data)

        list(_executor(threads).map(work, range(0, n, step)))
        return out


def _native_kernel():
    from remex import _native
    return _native.kernel()
