"""remex._native — compiled kernels for the encode path.

Two of them: the randomized-Hadamard apply behind
:class:`remex.rotation.RHTOperator`, and the code assignment that replaces
``np.searchsorted`` in :func:`remex.codebook.assign_codes`.

A small C library, compiled on first use with the system C compiler
(``cc``/``gcc``/``clang``), cached in a user-private directory and loaded
through :mod:`ctypes`. If no compiler is available, or ``REMEX_NO_NATIVE=1``
is set, ``kernel()`` returns ``None`` and :mod:`remex.rotation` uses its
NumPy implementation, which performs the same floating-point operations in
the same order and therefore produces the same bits.

Why the output is machine-independent: every step is a gather, an
elementwise multiply, or a pairwise add/subtract, each of which IEEE 754
rounds exactly. Nothing here depends on the instruction set or on how many
threads run it. ``-ffp-contract=off`` stops the compiler fusing a multiply
into an add, and no ``-march`` flag is used, so a library in a shared cache
cannot hit an illegal instruction on an older CPU.

The cache layout follows ``remax._native`` (CWE-379/367 hardening): a
user-private directory, a source-hash file name, write-to-temp then atomic
rename.

Measured against dense BLAS on x86, ARM and Apple Silicon:
``oaustegard/experiments`` ``rht-operator-native/RESULTS.md``.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

__all__ = ["kernel", "disable_reason"]

logger = logging.getLogger(__name__)

_C_SOURCE = r"""
#include <stdint.h>
#include <string.h>

/*
 * Randomized Hadamard rotation in operator form, the plan of
 * remex.rotation.rht_rotation: `rounds` rounds of
 * (permute -> sign * 1/sqrt(B) -> unnormalized block FWHT).
 *
 *   perms: (rounds, d) int32      ss: (rounds, d) sign / sqrt(B)
 *   mode 0: out = X @ R    (decode)   mode 1: out = X @ R.T (encode, query)
 *   buf:   caller-provided scratch of length d
 *
 * Rows are independent; callers split [lo, hi) across threads.
 */
#define DEFINE_RHT(T, NAME)                                                  \
static void fwht_##NAME(T *x, int64_t d, int64_t B) {                        \
    for (int64_t o = 0; o < d; o += B) {                                     \
        T *y = x + o;                                                        \
        for (int64_t h = 1; h < B; h <<= 1)                                  \
            for (int64_t i = 0; i < B; i += 2 * h)                           \
                for (int64_t j = i; j < i + h; j++) {                        \
                    T a = y[j], b = y[j + h];                                \
                    y[j] = a + b;                                            \
                    y[j + h] = a - b;                                        \
                }                                                            \
    }                                                                        \
}                                                                            \
void rht_apply_##NAME(const T *X, T *out, int64_t lo, int64_t hi,            \
                      int64_t d, int64_t B, int64_t rounds,                  \
                      const int32_t *perms, const T *ss, int32_t mode,       \
                      T *buf) {                                              \
    for (int64_t i = lo; i < hi; i++) {                                      \
        const T *src = X + i * d;                                            \
        T *dst = out + i * d;                                                \
        if (mode == 0) {                                                     \
            for (int64_t r = 0; r < rounds; r++) {                           \
                const int32_t *p = perms + r * d;                            \
                const T *s = ss + r * d;                                     \
                const T *in = src;                                           \
                if (r > 0) { memcpy(buf, dst, (size_t)d * sizeof(T)); in = buf; } \
                for (int64_t j = 0; j < d; j++) dst[j] = in[p[j]] * s[j];    \
                fwht_##NAME(dst, d, B);                                      \
            }                                                                \
        } else {                                                             \
            memcpy(dst, src, (size_t)d * sizeof(T));                         \
            for (int64_t r = rounds - 1; r >= 0; r--) {                      \
                const int32_t *p = perms + r * d;                            \
                const T *s = ss + r * d;                                     \
                fwht_##NAME(dst, d, B);                                      \
                memcpy(buf, dst, (size_t)d * sizeof(T));                     \
                for (int64_t j = 0; j < d; j++) dst[p[j]] = buf[j] * s[j];   \
            }                                                                \
        }                                                                    \
    }                                                                        \
}

DEFINE_RHT(float, f32)
DEFINE_RHT(double, f64)

/*
 * Code assignment: out[k] = #{b_i < x[k]}, which is exactly
 * np.searchsorted(b, x, side="left"), with NaN sent to the top cell as NumPy
 * does (NaN sorts above every boundary). Branchless binary search over the
 * 2^bits - 1 boundaries. Comparisons are exact in IEEE 754, so the NumPy and
 * compiled paths agree by construction, not by tolerance.
 */
#define DEFINE_QUANT(T, NAME)                                                \
void quant_##NAME(const T *x, uint8_t *out, int64_t lo, int64_t hi,          \
                  const T *b, int32_t bits) {                                \
    const int32_t m = (1 << bits) - 1;                                       \
    for (int64_t k = lo; k < hi; k++) {                                      \
        const T v = x[k];                                                    \
        int32_t idx = 0;                                                     \
        for (int32_t step = 1 << (bits - 1); step > 0; step >>= 1) {         \
            const int32_t c = idx + step;                                    \
            idx = (c <= m && v > b[c - 1]) ? c : idx;                        \
        }                                                                    \
        out[k] = (uint8_t)((v != v) ? m : idx);                              \
    }                                                                        \
}

DEFINE_QUANT(float, q32)
DEFINE_QUANT(double, q64)
"""

_FLAGS = ["-O3", "-ffp-contract=off", "-fPIC", "-shared"]
_SOURCE_HASH = hashlib.sha256(
    (_C_SOURCE + " ".join(_FLAGS)).encode()
).hexdigest()[:16]

_lock = threading.Lock()
_loaded = False
_kernel: Optional["_Kernel"] = None
_reason: Optional[str] = None


def _cache_dir() -> Path:
    """User-private cache: REMEX_CACHE_DIR, $XDG_CACHE_HOME/remex, ~/.cache/remex."""
    base = os.environ.get("REMEX_CACHE_DIR")
    if base:
        d = Path(base)
    elif os.environ.get("XDG_CACHE_HOME"):
        d = Path(os.environ["XDG_CACHE_HOME"]) / "remex"
    else:
        try:
            d = Path.home() / ".cache" / "remex"
        except (RuntimeError, KeyError):
            uid = os.getuid() if hasattr(os, "getuid") else 0
            d = Path(tempfile.gettempdir()) / f"remex_native_{uid}"
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def _compiler() -> Optional[str]:
    for name in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if name and shutil.which(name):
            return name
    return None


def _build() -> Path:
    cache = _cache_dir()
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    lib = cache / f"remex_rht_{_SOURCE_HASH}{suffix}"
    if lib.exists():
        return lib
    cc = _compiler()
    if cc is None:
        raise RuntimeError("no C compiler found (tried $CC, cc, gcc, clang)")
    fd, src = tempfile.mkstemp(suffix=".c", dir=cache)
    with os.fdopen(fd, "w") as f:
        f.write(_C_SOURCE)
    fd, tmp = tempfile.mkstemp(suffix=suffix, dir=cache)
    os.close(fd)
    try:
        proc = subprocess.run([cc, *_FLAGS, "-o", tmp, src],
                              capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"{cc} failed: {proc.stderr.strip()[:500]}")
        os.replace(tmp, lib)
    finally:
        for p in (src, tmp):
            try:
                os.unlink(p)
            except OSError:
                pass
    return lib


class _Kernel:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))  # CDLL releases the GIL per call
        vp, i64, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32
        argtypes = [vp, vp, i64, i64, i64, i64, i64, vp, vp, i32, vp]
        self.f32 = self.lib.rht_apply_f32
        self.f64 = self.lib.rht_apply_f64
        for fn in (self.f32, self.f64):
            fn.restype = None
            fn.argtypes = argtypes
        self.quant_f32 = self.lib.quant_q32
        self.quant_f64 = self.lib.quant_q64
        for fn in (self.quant_f32, self.quant_f64):
            fn.restype = None
            fn.argtypes = [vp, vp, i64, i64, vp, i32]
        self.path = path


def kernel() -> Optional[_Kernel]:
    """The loaded kernel, or ``None`` when the NumPy path must be used."""
    global _loaded, _kernel, _reason
    if _loaded:
        return _kernel
    with _lock:
        if _loaded:
            return _kernel
        if os.environ.get("REMEX_NO_NATIVE", "") not in ("", "0"):
            _reason = "disabled by REMEX_NO_NATIVE"
        else:
            try:
                _kernel = _Kernel(_build())
            except Exception as exc:  # noqa: BLE001 — any failure means fallback
                _reason = str(exc)
                logger.warning(
                    "remex: compiled RHT kernel unavailable (%s); using the "
                    "NumPy implementation, which gives identical codes but "
                    "is several times slower on large batches.", exc)
        _loaded = True
        return _kernel


def disable_reason() -> Optional[str]:
    """Why ``kernel()`` returned ``None``, if it did."""
    kernel()
    return _reason
