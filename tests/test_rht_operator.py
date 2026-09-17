"""RHT operator form (remex.rotation.RHTOperator) and machine-independent codebooks.

The operator replaces ``X @ R.T`` for ``rotation="rht"``. What must hold:
it is the same transform as ``rht_rotation`` (to float32 rounding), the
compiled and NumPy paths agree bit for bit, threading does not change a bit,
and the Quantizer routes through it without ever materializing R.
"""
import hashlib
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from remex import Quantizer
from remex import _native
from remex import rotation as rot
from remex.codebook import lloyd_max_codebook
from remex.rotation import RHTOperator, rht_rotation

DIMS = [6, 12, 100, 384, 768, 1024, 1536]
native = pytest.mark.skipif(_native.kernel() is None,
                            reason=f"no compiled kernel: {_native.disable_reason()}")


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


@pytest.fixture
def small_parallel_threshold(monkeypatch):
    monkeypatch.setattr(rot, "PARALLEL_MIN_FLOATS", 1)
    yield
    rot.set_num_threads(None)


@pytest.mark.parametrize("d", DIMS)
def test_operator_is_the_materialized_transform(d):
    R = rht_rotation(d, seed=11)
    op = RHTOperator(d, seed=11)
    X = np.random.default_rng(d).standard_normal((40, d)).astype(np.float32)
    np.testing.assert_allclose(op.rotate_rows(X), X @ R.T, atol=2e-5)
    np.testing.assert_allclose(op.unrotate_rows(X), X @ R, atol=2e-5)
    np.testing.assert_allclose(op.rotate_query(X[0]), R @ X[0], atol=2e-5)
    np.testing.assert_allclose(op.materialize(), R, atol=2e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_is_preserved_and_round_trips(dtype):
    op = RHTOperator(768, seed=3)
    X = np.random.default_rng(0).standard_normal((10, 768)).astype(dtype)
    Y = op.rotate_rows(X)
    assert Y.dtype == dtype
    tol = 1e-5 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(op.unrotate_rows(Y), X, atol=tol)


def test_odd_dimension_rejected():
    with pytest.raises(ValueError, match="odd"):
        RHTOperator(255)


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="Expected d=64"):
        RHTOperator(64).rotate_rows(np.zeros((2, 63), np.float32))


@native
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("d", DIMS)
def test_compiled_and_numpy_paths_agree_bit_for_bit(d, dtype):
    """The NumPy fallback is not an approximation of the kernel: same bits."""
    op = RHTOperator(d, seed=5)
    rng = np.random.default_rng(1)
    X = (rng.standard_normal((50, d)) * rng.lognormal(0, 3, (50, 1))).astype(dtype)
    for mode in (0, 1):
        assert np.array_equal(op._apply_native(_native.kernel(), X, mode),
                              op._apply_numpy(X, mode))


@native
@pytest.mark.parametrize("threads", [2, 3, 7])
def test_threading_does_not_change_a_bit(threads, small_parallel_threshold):
    op = RHTOperator(1536, seed=9)
    X = np.random.default_rng(2).standard_normal((101, 1536)).astype(np.float32)
    rot.set_num_threads(1)
    serial = (op.rotate_rows(X), op.unrotate_rows(X))
    rot.set_num_threads(threads)
    parallel = (op.rotate_rows(X), op.unrotate_rows(X))
    assert all(np.array_equal(a, b) for a, b in zip(serial, parallel))


def test_numpy_fallback_gives_the_same_codes():
    """REMEX_NO_NATIVE=1 in a fresh interpreter must reproduce this one's codes."""
    code = textwrap.dedent("""
        import hashlib, numpy as np
        from remex import Quantizer, _native
        X = np.random.default_rng(4).standard_normal((300, 512)).astype(np.float32)
        q = Quantizer(512, 4, seed=8, rotation="rht")
        print(_native.kernel() is None,
              hashlib.sha256(q.encode(X).indices.tobytes()).hexdigest())
    """)
    env = dict(os.environ, REMEX_NO_NATIVE="1")
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, check=True).stdout.split()
    assert out[0] == "True"
    X = np.random.default_rng(4).standard_normal((300, 512)).astype(np.float32)
    assert out[1] == _sha(Quantizer(512, 4, seed=8, rotation="rht").encode(X).indices)


def test_quantizer_routes_rht_through_the_operator():
    q = Quantizer(768, 4, seed=42, rotation="rht")
    X = np.random.default_rng(0).standard_normal((20, 768)).astype(np.float32)
    comp = q.encode(X)
    q.search(comp, X[0], k=5)
    q.decode(comp)
    assert q._R is None, "encode/search/decode materialized R"
    assert np.array_equal(q.R, rht_rotation(768, 42))


def test_assigning_R_replaces_the_operator():
    q = Quantizer(64, 4, seed=1, rotation="rht")
    q.R = np.eye(64, dtype=np.float32)
    X = np.random.default_rng(0).standard_normal((3, 64)).astype(np.float32)
    np.testing.assert_array_equal(q._rotate_rows(X), X)


def test_operator_codes_match_dense_codes_almost_everywhere():
    """Operator and dense R differ by float32 rounding, so codes can differ
    only for coordinates within ~1e-6 of a boundary."""
    d, bits = 768, 4
    X = np.random.default_rng(3).standard_normal((2000, d)).astype(np.float32)
    q = Quantizer(d, bits, seed=42, rotation="rht")
    op_codes = q.encode(X).indices
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    dense_codes = np.searchsorted(q.boundaries, U @ q.R.T).astype(np.uint8)
    assert (op_codes != dense_codes).mean() < 1e-4


@pytest.mark.skipif(platform.system() == "Windows" or not hasattr(os, "fork"),
                    reason="fork start method only")
def test_forked_child_can_encode_after_parent_used_threads(small_parallel_threshold):
    rot.set_num_threads(2)
    op = RHTOperator(256, seed=1)
    X = np.random.default_rng(0).standard_normal((64, 256)).astype(np.float32)
    expect = op.rotate_rows(X)  # parent pool now exists
    ctx = mp.get_context("fork")
    with ctx.Pool(1) as pool:
        res = pool.apply_async(_child_rotate, (X,))
        got = res.get(timeout=30)  # a stale pool in the child would hang here
    assert np.array_equal(got, expect)


def _child_rotate(X):
    rot.PARALLEL_MIN_FLOATS = 1
    rot.set_num_threads(2)
    return RHTOperator(256, seed=1).rotate_rows(X)


# ── codebook boundaries ──────────────────────────────────────────────────

@pytest.mark.parametrize("bits", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("d", [16, 384, 768, 3072])
def test_boundaries_are_exact_midpoints_of_the_float32_table(d, bits):
    b, c = lloyd_max_codebook(d, bits)
    c64 = c.astype(np.float64)
    assert b.dtype == np.float32 and c.dtype == np.float32
    assert np.array_equal(b, ((c64[:-1] + c64[1:]) / 2).astype(np.float32))
    assert b[(len(b) - 1) // 2] == 0.0 or bits == 1 and b[0] == 0.0


_X86_V4 = platform.machine().lower() in ("x86_64", "amd64")


@pytest.mark.skipif(not _X86_V4, reason="x86-64 dispatch levels only")
def test_boundaries_do_not_depend_on_numpy_simd_level():
    """NumPy's AVX-512 and AVX2 loops give scipy.stats.norm ulp-different
    values; the boundaries must not inherit that."""
    code = ("import hashlib; from remex.codebook import lloyd_max_codebook as f;"
            "print(hashlib.sha256(b''.join(f(d, b)[0].tobytes() "
            "for d in (384, 768, 3072) for b in (2, 4, 8))).hexdigest())")
    env0 = dict(os.environ)
    env1 = dict(os.environ, NPY_DISABLE_CPU_FEATURES="X86_V4")
    runs = [subprocess.run([sys.executable, "-c", code], env=e, capture_output=True,
                           text=True) for e in (env0, env1)]
    if runs[1].returncode != 0:
        pytest.skip(f"NumPy refused the feature mask: {runs[1].stderr[-200:]}")
    assert runs[0].stdout == runs[1].stdout
