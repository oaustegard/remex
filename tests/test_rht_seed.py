"""The seed must reach the decode for rotation="rht" at every d (issue #89).

Before 1.0, a power-of-two d took one round, and the rotate path was a fixed
Walsh-Hadamard transform followed by a seed-dependent signed permutation. The
Lloyd-Max codebook is identical for every coordinate and symmetric about
zero, so that permutation dropped out of decode: every seed gave the same
bits. ``R`` still differed by seed, which is why comparing matrices (as
``test_rotation_rht.test_deterministic_from_seed`` does) never caught it.
These tests compare decodes.
"""

import hashlib

import numpy as np
import pytest

from remex.core import Quantizer
from remex.rotation import RHTOperator, rht_plan, rht_rotation

POW2 = [2, 64, 128, 256, 1024]
NOT_POW2 = [6, 36, 384, 768]
SEEDS = (0, 1, 2)


def _unit_rows(n, d, seed=0):
    X = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


@pytest.mark.parametrize("d", POW2[1:])
def test_decode_depends_on_the_seed(d):
    """The issue's reproduction, inverted: seeds must not decode alike.

    refuted: with rht_rounds returning 1 when B == d (the pre-1.0 rule) this
    goes red at every d here.
    """
    X = _unit_rows(500, d)
    D = []
    for s in SEEDS:
        q = Quantizer(d, bits=2, seed=s, rotation="rht")
        D.append(q.decode(q.encode(X)))
    for other in D[1:]:
        assert not np.array_equal(D[0], other)


@pytest.mark.parametrize("d", POW2 + NOT_POW2)
def test_at_least_two_rounds_everywhere(d):
    assert rht_plan(d, 0)[1] >= 2


@pytest.mark.parametrize("d", POW2[1:])
def test_a_walsh_row_comes_out_spread(d):
    """A Walsh-Hadamard row is the input one round leaves one-hot."""
    # A single FWHT block over the whole row: its rows are Walsh rows.
    H = np.ones((1, 1), np.float32)
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]])
    W = H[[1, 3, d // 2, d - 1]] / np.float32(np.sqrt(d))
    for s in SEEDS:
        y = RHTOperator(d, s).rotate_rows(W)
        assert np.max(np.abs(y)) < 0.5, f"seed {s}: max |y| = {np.max(np.abs(y))}"


@pytest.mark.parametrize("d", [384, 768])
def test_codes_away_from_powers_of_two_did_not_move(d):
    """sha256 of "rht" codes as v0.8.0 wrote them. Only power-of-two d changed."""
    expected = {(384, 0): "126030fe998433ea", (384, 7): "336b535c40682f66",
                (768, 0): "6f54702179c7a41f", (768, 7): "2b93d98abc69055f"}
    X = np.random.default_rng(0).standard_normal((200, d)).astype(np.float32)
    for s in (0, 7):
        cv = Quantizer(d, bits=4, seed=s, rotation="rht").encode(X)
        got = hashlib.sha256(cv.indices.tobytes()).hexdigest()[:16]
        assert got == expected[(d, s)]


@pytest.mark.parametrize("d", [64, 256])
def test_operator_is_its_materialized_matrix_at_power_of_two_d(d):
    R = rht_rotation(d, seed=11)
    op = RHTOperator(d, seed=11)
    X = np.random.default_rng(d).standard_normal((40, d)).astype(np.float32)
    np.testing.assert_allclose(op.rotate_rows(X), X @ R.T, atol=2e-5)
    np.testing.assert_allclose(op.unrotate_rows(X), X @ R, atol=2e-5)


def test_default_is_rht_for_even_d_and_haar_for_odd_d():
    assert Quantizer(64, bits=2).rotation == "rht"
    assert Quantizer(384, bits=2).rotation == "rht"
    assert Quantizer(15, bits=2).rotation == "haar"
    X = _unit_rows(10, 64)
    assert Quantizer(64, bits=2).encode(X).rotation == "rht"
