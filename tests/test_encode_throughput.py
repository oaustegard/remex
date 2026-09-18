"""Compiled code assignment and block-wise encode.

Both changes must be invisible in the output: assign_codes reproduces
np.searchsorted exactly, and the encode block size changes only peak memory.
"""
import os
import subprocess
import sys
import tracemalloc
import textwrap

import numpy as np
import pytest

import remex.core as core
from remex import Quantizer, _native
from remex.codebook import assign_codes, lloyd_max_codebook

BITS = [1, 2, 3, 4, 8]


def _awkward(n, d, bits, boundaries, dtype):
    rng = np.random.default_rng(bits)
    X = (rng.standard_normal((n, d)) * rng.lognormal(0, 3, (n, 1))).astype(dtype)
    X[0, :4] = np.nan
    X[1, :min(d, len(boundaries))] = boundaries[:d]   # exact ties
    X[2, :2] = [np.inf, -np.inf]
    X[3, :2] = [np.finfo(np.float32).tiny, -0.0]
    return X


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("bits", BITS)
def test_assign_codes_matches_searchsorted_exactly(bits, dtype):
    b, _ = lloyd_max_codebook(256, bits)
    X = _awkward(200, 256, bits, b, dtype)
    assert np.array_equal(assign_codes(X, b, bits),
                          np.searchsorted(b, X).astype(np.uint8))


@pytest.mark.parametrize("bits", BITS)
def test_assign_codes_numpy_fallback_agrees(bits, monkeypatch):
    b, _ = lloyd_max_codebook(128, bits)
    X = _awkward(50, 128, bits, b, np.float32)
    compiled = assign_codes(X, b, bits)
    monkeypatch.setattr(_native, "kernel", lambda: None)
    assert np.array_equal(assign_codes(X, b, bits), compiled)


@pytest.mark.skipif(_native.kernel() is None, reason="no compiled kernel")
@pytest.mark.parametrize("threads", [1, 4])
def test_assign_codes_threading_is_invisible(threads):
    from remex import rotation as rot
    b, _ = lloyd_max_codebook(512, 4)
    X = _awkward(400, 512, 4, b, np.float32)
    rot.set_num_threads(1)
    serial = assign_codes(X, b, 4)
    rot.set_num_threads(threads)
    try:
        assert np.array_equal(assign_codes(X, b, 4), serial)
    finally:
        rot.set_num_threads(None)


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("rotation", ["rht", "haar", "none"])
def test_encode_does_not_depend_on_block_size(bits, centered, rotation):
    """Exactly invariant for operator and identity rotations. With
    rotation="haar" the unrotate inside centred-mode norms is a BLAS matmul
    whose rounding depends on the block shape, so only near-invariance is
    claimed there."""
    d, n = 96, 500
    X = np.random.default_rng(1).standard_normal((n, d)).astype(np.float32)
    kw = {"mean": X.mean(0)} if centered else {}
    q = Quantizer(d, bits, seed=2, rotation=rotation, **kw)
    full = q.encode(X)
    for block in (d, d * 7, d * n * 4):
        old = core.ENCODE_BLOCK_FLOATS
        core.ENCODE_BLOCK_FLOATS = block
        try:
            part = q.encode(X)
        finally:
            core.ENCODE_BLOCK_FLOATS = old
        if rotation == "haar" and centered:
            assert (part.indices != full.indices).mean() < 1e-4, block
            np.testing.assert_allclose(part.norms, full.norms, rtol=1e-5)
        else:
            assert np.array_equal(part.indices, full.indices), block
            assert np.array_equal(part.norms, full.norms), block


def test_scalar_encode_does_not_depend_on_block_size():
    d, n = 64, 300
    X = (np.random.default_rng(4).standard_normal((n, d)) * 1e5).astype(np.float64)
    X[0, 0] = np.nan
    q = Quantizer(d, 4, seed=1, rotation="rht", normalize=False)
    full = q.encode(X)
    old = core.ENCODE_BLOCK_FLOATS
    core.ENCODE_BLOCK_FLOATS = d * 3
    try:
        assert np.array_equal(q.encode(X).indices, full.indices)
    finally:
        core.ENCODE_BLOCK_FLOATS = old


def test_encode_peak_memory_is_bounded_by_the_block():
    """Peak allocation must scale with the block, not with the input."""
    d, n = 1024, 4000
    X = np.random.default_rng(0).standard_normal((n, d)).astype(np.float32)
    q = Quantizer(d, 4, seed=1, rotation="rht")
    old = core.ENCODE_BLOCK_FLOATS
    core.ENCODE_BLOCK_FLOATS = 1 << 18  # 256 rows here
    try:
        tracemalloc.start()
        q.encode(X)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    finally:
        core.ENCODE_BLOCK_FLOATS = old
    codes = n * d  # bytes of output
    assert peak < 3 * codes, f"peak {peak / 1e6:.0f} MB for {codes / 1e6:.0f} MB of codes"


def test_env_var_sets_the_block(tmp_path):
    code = textwrap.dedent("""
        import remex.core as core
        print(core.ENCODE_BLOCK_FLOATS)
    """)
    env = dict(os.environ, REMEX_ENCODE_BLOCK="4096")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "4096"
