"""The seed must reach the decode for rotation="rht2" at every d (issue #89).

At a power-of-two d, "rht" takes one round, and its rotate path is a fixed
Walsh-Hadamard transform followed by a seed-dependent signed permutation. The
Lloyd-Max codebook is identical for every coordinate and symmetric about
zero, so that permutation drops out of decode: every seed gives the same
bits. ``R`` still differs by seed, which is why comparing matrices (as
``test_rotation_rht.test_deterministic_from_seed`` does) never caught it.
These tests compare decodes.

"rht" stays as it shipped so files it wrote keep decoding; "rht2" is the fix.
"""

import hashlib
import warnings

import numpy as np
import pytest

from remex import CompressedVectors, load_pq, save_pq
from remex import _native
from remex.core import Quantizer
from remex.rotation import (
    RHT_MIN_ROUNDS, RHTOperator, rht2_rotation, rht_plan, rht_rotation,
)

POW2 = [64, 128, 256, 1024]
NOT_POW2 = [6, 384, 768]
SEEDS = (0, 1, 2)
native = pytest.mark.skipif(_native.kernel() is None,
                            reason=f"no compiled kernel: {_native.disable_reason()}")


def _unit_rows(n, d, seed=0):
    X = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _decodes(d, rotation, seeds=SEEDS):
    X = _unit_rows(500, d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        qs = [Quantizer(d, bits=2, seed=s, rotation=rotation) for s in seeds]
    return [q.decode(q.encode(X)) for q in qs]


@pytest.mark.parametrize("d", POW2)
def test_rht2_decode_depends_on_the_seed(d):
    """The issue's reproduction, inverted: seeds must not decode alike.

    refuted: with RHT_MIN_ROUNDS["rht2"] set to 1 this goes red at every d
    here, the same way "rht" does below.
    """
    D = _decodes(d, "rht2")
    for other in D[1:]:
        assert not np.array_equal(D[0], other)


@pytest.mark.parametrize("d", POW2)
def test_rht_at_power_of_two_d_is_still_seed_blind(d):
    """The defect, pinned as the frozen behaviour of the legacy name.

    If this goes red, "rht" changed transform and every file it wrote at a
    power-of-two d now decodes under a different basis.
    """
    D = _decodes(d, "rht")
    for other in D[1:]:
        assert np.array_equal(D[0], other)


@pytest.mark.parametrize("d", POW2)
def test_rht2_spreads_a_walsh_row(d):
    """A Walsh-Hadamard row is the input "rht" leaves one-hot at every seed."""
    H = rht_rotation(d, seed=0)          # rows are signed, permuted WHT rows
    x = H[:4]                            # four unit Walsh-aligned inputs
    for s in SEEDS:
        y_rht = RHTOperator(d, s, RHT_MIN_ROUNDS["rht"]).rotate_rows(x)
        y_rht2 = RHTOperator(d, s, RHT_MIN_ROUNDS["rht2"]).rotate_rows(x)
        assert np.allclose(np.max(np.abs(y_rht), axis=1), 1.0, atol=1e-5)
        assert np.max(np.abs(y_rht2)) < 0.5


@pytest.mark.parametrize("d", NOT_POW2)
def test_rht2_is_rht_away_from_powers_of_two(d):
    """Only power-of-two d changes. Everywhere else the two names encode alike."""
    assert rht_plan(d, 3, 1)[1] == rht_plan(d, 3, 2)[1] >= 2
    assert np.array_equal(rht_rotation(d, 3), rht2_rotation(d, 3))
    X = _unit_rows(50, d)
    a = Quantizer(d, bits=4, seed=3, rotation="rht").encode(X)
    b = Quantizer(d, bits=4, seed=3, rotation="rht2").encode(X)
    assert np.array_equal(a.indices, b.indices)


@pytest.mark.parametrize("d", [64, 384])
def test_rht_codes_did_not_move(d):
    """sha256 of "rht" codes as v0.8.0 wrote them, before rht2 existed."""
    expected = {(64, 0): "96663429488eb581", (64, 7): "38c4231430cebf77",
                (384, 0): "126030fe998433ea", (384, 7): "336b535c40682f66"}
    X = np.random.default_rng(0).standard_normal((200, d)).astype(np.float32)
    for s in (0, 7):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cv = Quantizer(d, bits=4, seed=s, rotation="rht").encode(X)
        got = hashlib.sha256(cv.indices.tobytes()).hexdigest()[:16]
        assert got == expected[(d, s)]


@pytest.mark.parametrize("d", [64, 256, 384])
def test_rht2_operator_is_its_materialized_matrix(d):
    R = rht2_rotation(d, seed=11)
    op = RHTOperator(d, seed=11, min_rounds=RHT_MIN_ROUNDS["rht2"])
    X = np.random.default_rng(d).standard_normal((40, d)).astype(np.float32)
    np.testing.assert_allclose(op.rotate_rows(X), X @ R.T, atol=2e-5)
    np.testing.assert_allclose(op.unrotate_rows(X), X @ R, atol=2e-5)
    q = Quantizer(d, bits=4, seed=11, rotation="rht2")
    np.testing.assert_allclose(q.R, R, atol=0)


@native
@pytest.mark.parametrize("d", [64, 1024])
def test_rht2_compiled_and_numpy_paths_agree_bit_for_bit(d):
    op = RHTOperator(d, seed=5, min_rounds=2)
    X = np.random.default_rng(1).standard_normal((50, d)).astype(np.float32)
    for mode in (0, 1):
        assert np.array_equal(op._apply_native(_native.kernel(), X, mode),
                              op._apply_numpy(X, mode))


def test_rht_warns_only_where_it_is_seed_blind():
    with pytest.warns(UserWarning, match="rht2"):
        Quantizer(64, bits=2, rotation="rht")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Quantizer(384, bits=2, rotation="rht")
        Quantizer(64, bits=2, rotation="rht2")


def test_rht2_files_refuse_an_rht_decoder(tmp_path):
    X = _unit_rows(20, 64)
    cv = Quantizer(64, bits=4, seed=1, rotation="rht2").encode(X)
    save_pq(tmp_path / "a.pq", cv)
    cv.save(tmp_path / "a.npz")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        legacy = Quantizer(64, bits=4, seed=1, rotation="rht")
    for loaded in (load_pq(tmp_path / "a.pq"),
                   CompressedVectors.load(tmp_path / "a.npz")):
        assert loaded.rotation == "rht2"
        with pytest.raises(ValueError):
            legacy.decode(loaded)
