"""remex.anisotropy and the AnisotropyWarning an uncentered encode raises.

The warning is advice only. The contract under test is that it fires on the
corpora where centered mode was measured to pay (a large shared direction,
at <= 4 bits), stays silent everywhere else, and never changes the codes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import remex
from remex import AnisotropyWarning, Quantizer, anisotropy, corpus_mean
from remex.core import ANISOTROPY_MIN_ROWS, ANISOTROPY_WARN_RATIO

D = 64


def _isotropic(n=2000, d=D, seed=0):
    X = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _cone(n=2000, d=D, seed=0, spread=0.3):
    """Unit vectors around one shared direction, like mxbai-edge-colbert tokens
    (corpus-mean norm 0.94)."""
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(d); mu /= np.linalg.norm(mu)
    X = mu + spread * rng.standard_normal((n, d)) / np.sqrt(d)
    X = X.astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _encode_catching(q, X):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cv = q.encode(X)
    return cv, [w for w in rec if issubclass(w.category, AnisotropyWarning)]


class TestAnisotropy:
    def test_isotropic_is_near_zero(self):
        assert anisotropy(_isotropic()) < 0.1

    def test_cone_is_near_one(self):
        assert anisotropy(_cone()) > 0.9

    def test_matches_definition(self):
        X = _cone(n=300, seed=3, spread=1.5)
        want = np.linalg.norm(X.astype(np.float64).mean(0)) / np.linalg.norm(X.astype(np.float64), axis=1).mean()
        assert anisotropy(X) == pytest.approx(want, rel=1e-12)

    def test_scale_invariant(self):
        X = _cone(n=300, spread=1.0)
        assert anisotropy(3.0 * X) == pytest.approx(anisotropy(X), rel=1e-9)

    @pytest.mark.parametrize("X", [np.ones((1, D)), np.zeros((5, D))])
    def test_degenerate_is_none(self, X):
        assert anisotropy(X) is None

    def test_rejects_1d(self):
        with pytest.raises(ValueError):
            anisotropy(np.ones(D))

    def test_exported(self):
        assert remex.anisotropy is anisotropy
        assert "anisotropy" in remex.__all__ and "AnisotropyWarning" in remex.__all__
        assert issubclass(AnisotropyWarning, UserWarning)


class TestEncodeWarns:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_warns_on_cone_at_low_bits(self, bits):
        _, got = _encode_catching(Quantizer(D, bits=bits, seed=0), _cone())
        assert len(got) == 1
        msg = str(got[0].message)
        assert "mean=remex.corpus_mean(X)" in msg
        assert f"{anisotropy(_cone()):.2f}" in msg

    def test_points_at_the_caller(self):
        _, got = _encode_catching(Quantizer(D, bits=2, seed=0), _cone())
        assert got[0].filename == __file__

    def test_ratio_matches_public_function(self):
        """encode computes the ratio blockwise; it must equal anisotropy()."""
        X = _cone(n=5000, spread=0.8)
        _, got = _encode_catching(Quantizer(D, bits=2, seed=0), X)
        assert f"{anisotropy(X):.2f}" in str(got[0].message)

    def test_codes_unchanged_by_the_check(self):
        X = _cone()
        cv, _ = _encode_catching(Quantizer(D, bits=2, seed=0), X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AnisotropyWarning)
            ref = Quantizer(D, bits=2, seed=0).encode(X)
        assert np.array_equal(cv.indices, ref.indices)
        assert np.array_equal(cv.norms, ref.norms)

    def test_filterable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings("ignore", category=AnisotropyWarning)
            Quantizer(D, bits=2, seed=0).encode(_cone())


class TestEncodeSilent:
    def test_isotropic(self):
        assert _encode_catching(Quantizer(D, bits=2, seed=0), _isotropic())[1] == []

    def test_centered_mode(self):
        X = _cone()
        assert _encode_catching(Quantizer(D, bits=2, seed=0, mean=corpus_mean(X)), X)[1] == []

    def test_eight_bit(self):
        assert _encode_catching(Quantizer(D, bits=8, seed=0), _cone())[1] == []

    def test_scalar_mode(self):
        q = Quantizer(D, bits=2, seed=0, normalize=False, rotation="none")
        assert _encode_catching(q, _cone())[1] == []

    def test_small_batch(self):
        X = _cone(n=ANISOTROPY_MIN_ROWS - 1)
        assert _encode_catching(Quantizer(D, bits=2, seed=0), X)[1] == []

    def test_single_query_row(self):
        assert _encode_catching(Quantizer(D, bits=2, seed=0), _cone(n=1))[1] == []

    def test_just_below_threshold(self):
        """Find a spread whose ratio sits just under the threshold and check silence."""
        for spread in np.linspace(0.5, 6.0, 111):
            X = _cone(spread=spread)
            r = anisotropy(X)
            if ANISOTROPY_WARN_RATIO - 0.05 < r < ANISOTROPY_WARN_RATIO:
                assert _encode_catching(Quantizer(D, bits=2, seed=0), X)[1] == []
                return
        pytest.fail("no spread landed just below the threshold")
