"""Tests filling coverage gaps (Issue #20).

Covers:
1. Edge cases: empty corpus, single vector, k > n
2. Save/load roundtrip at all supported bit widths
3. ADC + Matryoshka precision consistency
4. GPUSearcher two-stage score equality (not just shape)
5. CompressedVectors subset search correctness
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from remex import Quantizer, CompressedVectors


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ------------------------------------------------------------------
# 1. Edge cases: empty corpus, single vector, k > n
# ------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_corpus(self, rng):
        """n=0 corpus should encode and return empty results."""
        pq = Quantizer(d=64, bits=4)
        X_empty = np.empty((0, 64), dtype=np.float32)
        comp = pq.encode(X_empty)
        assert comp.n == 0
        assert comp.indices.shape == (0, 64)

    def test_single_vector_corpus(self, rng):
        """n=1 corpus should work for all search methods."""
        pq = Quantizer(d=64, bits=4)
        X = rng.standard_normal((1, 64)).astype(np.float32)
        query = rng.standard_normal(64).astype(np.float32)
        comp = pq.encode(X)

        idx, scores = pq.search(comp, query, k=1)
        assert idx.shape == (1,)
        assert idx[0] == 0

        idx_adc, scores_adc = pq.search_adc(comp, query, k=1)
        assert idx_adc.shape == (1,)
        assert idx_adc[0] == 0

        idx_ts, scores_ts = pq.search_twostage(
            comp, query, k=1, candidates=1
        )
        assert idx_ts.shape == (1,)
        assert idx_ts[0] == 0

    def test_k_exceeds_n_all_methods(self, rng):
        """k > n should return all n results for every search method."""
        pq = Quantizer(d=64, bits=4)
        X = rng.standard_normal((5, 64)).astype(np.float32)
        query = rng.standard_normal(64).astype(np.float32)
        comp = pq.encode(X)

        idx, scores = pq.search(comp, query, k=100)
        assert len(idx) == 5

        idx_adc, scores_adc = pq.search_adc(comp, query, k=100)
        assert len(idx_adc) == 5

        idx_ts, scores_ts = pq.search_twostage(
            comp, query, k=100, candidates=5
        )
        assert len(idx_ts) == 5


# ------------------------------------------------------------------
# 2. Save/load roundtrip at all supported bit widths
# ------------------------------------------------------------------

class TestSaveLoadAllBits:

    @pytest.mark.parametrize("bits", [1, 2, 3, 4, 8])
    def test_save_load_roundtrip(self, bits, tmp_path, rng):
        """save() / load() must roundtrip at every supported bit width."""
        pq = Quantizer(d=64, bits=bits)
        X = rng.standard_normal((50, 64)).astype(np.float32)
        comp = pq.encode(X)

        path = str(tmp_path / f"test_{bits}bit.npz")
        comp.save(path)
        loaded = CompressedVectors.load(path)

        np.testing.assert_array_equal(comp.indices, loaded.indices)
        np.testing.assert_allclose(comp.norms, loaded.norms)
        assert loaded.d == comp.d
        assert loaded.bits == comp.bits
        assert loaded.n == comp.n


# ------------------------------------------------------------------
# 3. ADC + Matryoshka precision: search_adc(precision=2) on 4-bit
# ------------------------------------------------------------------

class TestADCMatryoshkaPrecision:

    def test_adc_precision_matches_cached(self, rng):
        """search_adc(precision=2) on 4-bit corpus should match
        cached search at the same precision."""
        d = 128
        pq = Quantizer(d=d, bits=4)
        corpus = rng.standard_normal((500, d)).astype(np.float32)
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
        query = rng.standard_normal(d).astype(np.float32)
        query /= np.linalg.norm(query)

        comp = pq.encode(corpus)

        idx_cached, scores_cached = pq.search(comp, query, k=10, precision=2)
        # Clear cache so ADC doesn't use it
        comp.drop_cache()
        idx_adc, scores_adc = pq.search_adc(comp, query, k=10, precision=2)

        np.testing.assert_array_equal(idx_cached, idx_adc)
        np.testing.assert_allclose(scores_cached, scores_adc, rtol=1e-5)


# ------------------------------------------------------------------
# 4. GPUSearcher two-stage: score equality, not just shape
# ------------------------------------------------------------------

class TestGPUSearcherTwoStageScores:

    def test_twostage_scores_match_pq(self, rng):
        """GPUSearcher.search_twostage scores must match
        Quantizer.search_twostage exactly, not just shapes."""
        from remex.gpu import GPUSearcher

        d = 128
        pq = Quantizer(d=d, bits=4)
        corpus = rng.standard_normal((500, d)).astype(np.float32)
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
        query = rng.standard_normal(d).astype(np.float32)
        query /= np.linalg.norm(query)

        comp = pq.encode(corpus)
        searcher = GPUSearcher(pq, comp, backend="numpy")

        idx_pq, scores_pq = pq.search_twostage(
            comp, query, k=10, candidates=200
        )
        idx_gpu, scores_gpu = searcher.search_twostage(
            query, k=10, candidates=200
        )

        np.testing.assert_array_equal(idx_pq, idx_gpu)
        np.testing.assert_allclose(scores_pq, scores_gpu, rtol=1e-5)


# ------------------------------------------------------------------
# 5. CompressedVectors subset search: indices into subset, not original
# ------------------------------------------------------------------

class TestSubsetSearch:

    def test_subset_search_indices_local(self, rng):
        """After subsetting, search results should be indices into
        the subset, not the original corpus."""
        d = 64
        pq = Quantizer(d=d, bits=4)
        corpus = rng.standard_normal((200, d)).astype(np.float32)
        query = rng.standard_normal(d).astype(np.float32)

        comp = pq.encode(corpus)

        # Take a subset of 50 vectors
        subset_idx = rng.choice(200, size=50, replace=False)
        sub = comp.subset(subset_idx)

        assert sub.n == 50

        # Search the subset
        result_idx, result_scores = pq.search(sub, query, k=10)

        # Result indices must be in [0, 50)
        assert np.all(result_idx >= 0)
        assert np.all(result_idx < 50)

        # Scores should be descending
        assert np.all(np.diff(result_scores) <= 1e-7)

        # Verify the returned indices correspond to the right vectors
        # by checking that the top result's score is consistent
        best_idx = result_idx[0]
        # Decode the best vector from subset and verify inner product
        x_hat = pq.decode(sub)[best_idx]
        expected_score = np.dot(x_hat, query)
        np.testing.assert_allclose(
            result_scores[0], expected_score, rtol=1e-4
        )
