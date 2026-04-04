"""Tests for ADC search and GPU backend (numpy fallback).

Verifies:
- ADC search produces identical rankings to cached search
- search_twostage (ADC) matches old two-stage behavior
- GPUSearcher (numpy backend) matches Quantizer results
- Memory: ADC doesn't populate the float32 cache
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
def setup():
    rng = np.random.default_rng(42)
    d = 128
    n = 2000
    pq = Quantizer(d=d, bits=4)
    corpus = rng.standard_normal((n, d)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    query = rng.standard_normal(d).astype(np.float32)
    query /= np.linalg.norm(query)
    compressed = pq.encode(corpus)
    return pq, compressed, query, corpus


@pytest.fixture
def setup_8bit():
    """8-bit quantizer for two-stage tests."""
    rng = np.random.default_rng(42)
    d = 128
    n = 2000
    pq = Quantizer(d=d, bits=8)
    corpus = rng.standard_normal((n, d)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = [rng.standard_normal(d).astype(np.float32) for _ in range(5)]
    queries = [q / np.linalg.norm(q) for q in queries]
    compressed = pq.encode(corpus)
    return pq, compressed, queries


# ------------------------------------------------------------------
# ADC search tests
# ------------------------------------------------------------------

class TestADCSearch:

    def test_adc_matches_cached_topk(self, setup):
        """ADC top-k should match cached search top-k."""
        pq, comp, query, _ = setup
        idx_cached, scores_cached = pq.search(comp, query, k=10)
        idx_adc, scores_adc = pq.search_adc(comp, query, k=10)
        np.testing.assert_array_equal(idx_cached, idx_adc)
        np.testing.assert_allclose(scores_cached, scores_adc, rtol=1e-5)

    def test_adc_precision_parameter(self, setup):
        """ADC at reduced precision should differ from full."""
        pq, comp, query, _ = setup
        idx_full, _ = pq.search_adc(comp, query, k=10)
        idx_low, _ = pq.search_adc(comp, query, k=10, precision=2)
        assert not np.array_equal(idx_full, idx_low)

    def test_adc_no_cache_populated(self, setup):
        """ADC search must not populate the _x_hat_rot cache."""
        pq, comp, query, _ = setup
        assert comp._x_hat_rot is None
        pq.search_adc(comp, query, k=10)
        assert comp._x_hat_rot is None

    def test_adc_chunk_sizes(self, setup):
        """Different chunk sizes should produce identical results."""
        pq, comp, query, _ = setup
        idx_a, scores_a = pq.search_adc(comp, query, k=10, chunk_size=64)
        idx_b, scores_b = pq.search_adc(comp, query, k=10, chunk_size=4096)
        np.testing.assert_array_equal(idx_a, idx_b)
        np.testing.assert_allclose(scores_a, scores_b)

    def test_adc_scores_descending(self, setup):
        pq, comp, query, _ = setup
        _, scores = pq.search_adc(comp, query, k=20)
        assert np.all(np.diff(scores) <= 1e-7)

    def test_adc_k_larger_than_n(self, setup):
        pq, comp, query, _ = setup
        idx, scores = pq.search_adc(comp, query, k=comp.n + 100)
        assert len(idx) == comp.n


# ------------------------------------------------------------------
# Two-stage (ADC) tests
# ------------------------------------------------------------------

class TestTwoStageADC:

    def test_returns_correct_k(self, setup_8bit):
        pq, comp, queries = setup_8bit
        idx, scores = pq.search_twostage(comp, queries[0], k=10, candidates=200)
        assert len(idx) == 10
        assert len(scores) == 10

    def test_indices_in_range(self, setup_8bit):
        pq, comp, queries = setup_8bit
        idx, _ = pq.search_twostage(comp, queries[0], k=10, candidates=200)
        assert np.all(idx >= 0)
        assert np.all(idx < comp.n)

    def test_scores_descending(self, setup_8bit):
        pq, comp, queries = setup_8bit
        _, scores = pq.search_twostage(comp, queries[0], k=10, candidates=200)
        assert np.all(np.diff(scores) <= 1e-7)

    def test_large_candidates_matches_full(self, setup_8bit):
        """With candidates=n, two-stage should match single-stage."""
        pq, comp, queries = setup_8bit
        idx_full, _ = pq.search(comp, queries[0], k=10)
        idx_ts, _ = pq.search_twostage(
            comp, queries[0], k=10, candidates=comp.n
        )
        np.testing.assert_array_equal(idx_full, idx_ts)

    def test_no_cache_after_twostage(self, setup_8bit):
        """Two-stage ADC must not leave a float32 cache on the corpus."""
        pq, comp, queries = setup_8bit
        assert comp._x_hat_rot is None
        pq.search_twostage(comp, queries[0], k=10, candidates=200)
        assert comp._x_hat_rot is None

    def test_multiple_queries_consistent(self, setup_8bit):
        """Repeated queries should return identical results."""
        pq, comp, queries = setup_8bit
        q = queries[0]
        idx1, s1 = pq.search_twostage(comp, q, k=10, candidates=200)
        idx2, s2 = pq.search_twostage(comp, q, k=10, candidates=200)
        np.testing.assert_array_equal(idx1, idx2)
        np.testing.assert_allclose(s1, s2)


# ------------------------------------------------------------------
# Memory accounting tests
# ------------------------------------------------------------------

class TestMemoryAccounting:

    def test_resident_bytes_no_cache(self, setup):
        pq, comp, _, _ = setup
        base = comp.indices.nbytes + comp.norms.nbytes
        assert comp.resident_bytes == base

    def test_resident_bytes_with_cache(self, setup):
        pq, comp, query, _ = setup
        pq.search(comp, query, k=10)  # populates cache
        assert comp._x_hat_rot is not None
        expected = comp.indices.nbytes + comp.norms.nbytes + comp._x_hat_rot.nbytes
        assert comp.resident_bytes == expected

    def test_drop_cache(self, setup):
        pq, comp, query, _ = setup
        pq.search(comp, query, k=10)
        assert comp._x_hat_rot is not None
        comp.drop_cache()
        assert comp._x_hat_rot is None
        base = comp.indices.nbytes + comp.norms.nbytes
        assert comp.resident_bytes == base


# ------------------------------------------------------------------
# GPU backend tests (numpy fallback)
# ------------------------------------------------------------------

class TestGPUSearcherNumpy:
    """Test GPUSearcher using the numpy fallback backend.

    This exercises the full GPUSearcher code path without a GPU.
    Results must match Quantizer exactly.
    """

    @pytest.fixture
    def gpu_setup(self, setup):
        pq, comp, query, corpus = setup
        from remex.gpu import GPUSearcher
        searcher = GPUSearcher(pq, comp, backend="numpy")
        return pq, comp, query, corpus, searcher

    def test_backend_is_numpy(self, gpu_setup):
        _, _, _, _, searcher = gpu_setup
        assert searcher.backend_name == "numpy"

    def test_search_matches_pq(self, gpu_setup):
        pq, comp, query, _, searcher = gpu_setup
        idx_pq, scores_pq = pq.search(comp, query, k=10)
        idx_gpu, scores_gpu = searcher.search(query, k=10)
        np.testing.assert_array_equal(idx_pq, idx_gpu)
        np.testing.assert_allclose(scores_pq, scores_gpu, rtol=1e-5)

    def test_search_adc_matches_pq(self, gpu_setup):
        pq, comp, query, _, searcher = gpu_setup
        idx_pq, scores_pq = pq.search_adc(comp, query, k=10)
        idx_gpu, scores_gpu = searcher.search_adc(query, k=10)
        np.testing.assert_array_equal(idx_pq, idx_gpu)
        np.testing.assert_allclose(scores_pq, scores_gpu, rtol=1e-5)

    def test_twostage_matches_pq(self, gpu_setup):
        pq, comp, query, _, searcher = gpu_setup
        idx_pq, scores_pq = pq.search_twostage(
            comp, query, k=10, candidates=200
        )
        idx_gpu, scores_gpu = searcher.search_twostage(
            query, k=10, candidates=200
        )
        np.testing.assert_array_equal(idx_pq, idx_gpu)
        np.testing.assert_allclose(scores_pq, scores_gpu, rtol=1e-5)

    def test_drop_cache(self, gpu_setup):
        _, _, query, _, searcher = gpu_setup
        searcher.search(query, k=10)  # build cache
        assert searcher._x_hat_rot_gpu is not None
        searcher.drop_cache()
        assert searcher._x_hat_rot_gpu is None

    def test_available_backends_includes_numpy(self):
        from remex.gpu import available_backends
        assert "numpy" in available_backends()
