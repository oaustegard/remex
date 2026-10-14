"""Centered mode: encode the offset from a caller-declared corpus mean.

remex stays training-free by never measuring the mean itself. The caller
declares it, exactly as it declares ``scale`` in scalar mode, and
``remex.corpus_mean`` is the blessed way to compute one.

Each row stores ``||x - mu||`` in ``norms`` and ``mu . (x - mu)`` in
``mean_dots`` (one float32, 4 B per vector). ``decode`` returns
``w mu + kappa ||r|| u_hat`` with ``w`` making ``mu . (decode(x) - mu)`` exact.
Search does not score ``q . decode(x)``: it scores
``q . mu + mu . r + ||r|| (q - mu) . u_hat / kappa``, the estimator faiss's
RaBitQ uses, so the quantisation error is multiplied by ``||q - mu||`` rather
than ``||q||``. ``Quantizer._score_vectors`` returns the vectors and row bias
that reproduce those scores; the path-agreement tests check against it.
"""
import numpy as np
import pytest

import remex
from remex import Quantizer


def shifted_corpus(n=1200, d=96, shift=1.0, seed=0):
    """A corpus with a large common component, which is what centering targets."""
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(d).astype(np.float32) * shift
    return (mu + rng.standard_normal((n, d)).astype(np.float32)).astype(np.float32)


def topk(M, Q, k):
    return np.argsort(-(Q @ M.T), axis=1)[:, :k]


def score_ref(pq, c, q, precision=None):
    """The search estimate for every row, from ``_score_vectors``."""
    Y, bias = pq._score_vectors(c, precision)
    return Y @ q + (0.0 if bias is None else bias)


def recall(pred, truth, k):
    return sum(len(set(a[:k].tolist()) & set(b[:k].tolist()))
               for a, b in zip(pred, truth)) / (len(pred) * k)


class TestCorpusMean:
    def test_helper_matches_numpy(self):
        X = shifted_corpus()
        assert np.allclose(remex.corpus_mean(X), X.mean(axis=0), rtol=1e-5)

    def test_helper_returns_float32(self):
        assert remex.corpus_mean(shifted_corpus()).dtype == np.float32


class TestReconstruction:
    def test_mean_component_is_exact(self):
        """``mu . (decode(x) - mu)`` equals the stored ``mu . r``."""
        X = shifted_corpus()
        mu = remex.corpus_mean(X)
        pq = Quantizer(d=X.shape[1], bits=3, seed=42, mean=mu)
        c = pq.encode(X)
        got = (pq.decode(c).astype(np.float64) - mu) @ mu.astype(np.float64)
        assert np.allclose(got, c.mean_dots, rtol=1e-3, atol=1e-2)

    @pytest.mark.parametrize("shift", [0.3, 3.0, 10.0])
    def test_reconstruction_beats_restoring_the_residual_length(self, shift):
        """MSE no worse than ``mu + ||r|| u_hat``, at any shift, and finite."""
        X = shifted_corpus(shift=shift)
        pq = Quantizer(d=X.shape[1], bits=2, seed=42, mean=remex.corpus_mean(X))
        c = pq.encode(X)
        v = pq._unrotate_rows(pq.centroids[c.indices])
        plain = c.mean + c.norms[:, None] * v / np.linalg.norm(v, axis=1, keepdims=True)
        rec = pq.decode(c)
        assert np.all(np.isfinite(rec))
        assert np.mean((rec - X) ** 2) <= np.mean((plain - X) ** 2)

    def test_uncentered_reconstruction_is_unchanged(self):
        X = shifted_corpus()
        plain = Quantizer(d=X.shape[1], bits=4, seed=42)
        assert np.allclose(np.linalg.norm(plain.decode(plain.encode(X)), axis=1),
                           np.linalg.norm(X, axis=1), rtol=1e-3)

    def test_mean_is_carried_on_the_container(self):
        X = shifted_corpus()
        mu = remex.corpus_mean(X)
        c = Quantizer(d=X.shape[1], bits=4, seed=42, mean=mu).encode(X)
        assert c.mean is not None
        assert np.array_equal(c.mean, mu)

    def test_plain_container_has_no_mean(self):
        X = shifted_corpus()
        assert Quantizer(d=X.shape[1], bits=4, seed=42).encode(X).mean is None


class TestNestedPrecision:
    """A centred code read at a lower Matryoshka precision.

    The stored length ``m`` solves ``||mu + m*u_hat|| == ||x||`` for the
    full-precision direction. Reused unchanged with a coarser direction, the
    reconstruction lands at the wrong length; on pplx-embed-v2-late tokens
    (||mean|| / mean ||x|| = 0.945) that took a 1-bit read of an 8-bit code
    from nDCG@10 0.765 to 0.527. ``||x||`` is recoverable from the full code,
    so ``m`` is re-solved per precision with no stored bytes.
    """

    @pytest.mark.parametrize("p", [1, 2, 3, 4, 6, 7])
    def test_nested_decode_keeps_the_mean_component(self, p):
        X = shifted_corpus()
        mu = remex.corpus_mean(X)
        pq = Quantizer(d=X.shape[1], bits=8, seed=42, mean=mu)
        c = pq.encode(X)
        got = (pq.decode(c, precision=p).astype(np.float64) - mu) @ mu.astype(np.float64)
        assert np.allclose(got, c.mean_dots, rtol=1e-3, atol=1e-2)

    def test_nested_1bit_equals_standalone_1bit(self):
        """The MSB of a centred 8-bit code is the centred 1-bit code, and the
        two decode to the same vectors."""
        X = shifted_corpus(seed=5)
        mu = remex.corpus_mean(X)
        q8 = Quantizer(d=X.shape[1], bits=8, seed=42, mean=mu)
        q1 = Quantizer(d=X.shape[1], bits=1, seed=42, mean=mu)
        c8, c1 = q8.encode(X), q1.encode(X)
        assert np.array_equal(c8.indices >> 7, c1.indices)
        assert np.allclose(q8.decode(c8, precision=1), q1.decode(c1),
                           rtol=1e-4, atol=1e-5)

    def test_full_precision_is_untouched(self):
        X = shifted_corpus(seed=6)
        pq = Quantizer(d=X.shape[1], bits=4, seed=42, mean=remex.corpus_mean(X))
        c = pq.encode(X)
        assert np.array_equal(pq.decode(c), pq.decode(c, precision=4))

    @pytest.mark.parametrize("p", [1, 2])
    def test_nested_paths_agree(self, p):
        from remex import IVFCoarseIndex, PackedVectors
        from remex.gpu import GPUSearcher
        X = shifted_corpus(n=800, d=64, seed=13)
        pq = Quantizer(d=X.shape[1], bits=4, seed=42, mean=remex.corpus_mean(X))
        c = pq.encode(X)
        q = X[31]
        i_a, s_a = pq.search_adc(c, q, k=10, precision=p)
        assert np.allclose(s_a, score_ref(pq, c, q, p)[i_a], rtol=1e-3, atol=1e-4)
        i_p, s_p = pq.search_adc(PackedVectors.from_compressed(c), q, k=10, precision=p)
        assert np.array_equal(i_a, i_p) and np.allclose(s_a, s_p, rtol=1e-4, atol=1e-5)
        i_c, s_c = pq.search(c, q, k=10, precision=p)
        assert np.array_equal(i_a, i_c) and np.allclose(s_a, s_c, rtol=1e-4, atol=1e-4)
        i_i, s_i = IVFCoarseIndex(pq, c, n_bits=4).search_coarse(q, k=10, nprobe=16, precision=p)
        assert np.array_equal(i_a, i_i) and np.allclose(s_a, s_i, rtol=1e-4, atol=1e-4)
        i_g, s_g = GPUSearcher(pq, c, backend="numpy").search_adc(q, k=10, precision=p)
        assert np.array_equal(i_a, i_g) and np.allclose(s_a, s_g, rtol=1e-4, atol=1e-4)


class TestGuards:
    def test_decoding_centered_codes_without_the_mean_raises(self):
        X = shifted_corpus()
        c = Quantizer(d=X.shape[1], bits=4, seed=42, mean=remex.corpus_mean(X)).encode(X)
        with pytest.raises(ValueError, match="mean mismatch"):
            Quantizer(d=X.shape[1], bits=4, seed=42).decode(c)

    def test_decoding_plain_codes_with_a_mean_raises(self):
        X = shifted_corpus()
        c = Quantizer(d=X.shape[1], bits=4, seed=42).encode(X)
        with pytest.raises(ValueError, match="mean mismatch"):
            Quantizer(d=X.shape[1], bits=4, seed=42, mean=remex.corpus_mean(X)).decode(c)

    def test_a_different_mean_raises(self):
        X = shifted_corpus()
        mu = remex.corpus_mean(X)
        c = Quantizer(d=X.shape[1], bits=4, seed=42, mean=mu).encode(X)
        with pytest.raises(ValueError, match="mean mismatch"):
            Quantizer(d=X.shape[1], bits=4, seed=42, mean=mu + 1.0).decode(c)

    def test_pre_2_0_centred_container_is_refused(self):
        from remex import CompressedVectors
        X = shifted_corpus()
        pq = Quantizer(d=X.shape[1], bits=4, seed=42, mean=remex.corpus_mean(X))
        c = pq.encode(X)
        old = CompressedVectors(c.indices, c.norms, c.d, c.bits, c.rotation, c.mean)
        with pytest.raises(ValueError, match="remex < 2.0"):
            pq.decode(old)
        with pytest.raises(ValueError, match="remex < 2.0"):
            pq.search(old, X[0], k=5)

    def test_mean_rejected_in_scalar_mode(self):
        with pytest.raises(ValueError, match="normalize=True"):
            Quantizer(d=8, bits=4, normalize=False, scale=1.0, mean=np.zeros(8, np.float32))

    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match="shape"):
            Quantizer(d=8, bits=4, mean=np.zeros(9, np.float32))

    def test_non_finite_rejected(self):
        bad = np.zeros(8, np.float32); bad[0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            Quantizer(d=8, bits=4, mean=bad)


class TestSearchPaths:
    @pytest.mark.parametrize("bits", [2, 4])
    def test_adc_matches_cached(self, bits):
        X = shifted_corpus(n=700, d=64, seed=1)
        pq = Quantizer(d=X.shape[1], bits=bits, seed=42, mean=remex.corpus_mean(X))
        c = pq.encode(X)
        i_a, s_a = pq.search(c, X[5], k=10)
        i_b, s_b = pq.search_adc(c, X[5], k=10)
        assert np.array_equal(i_a, i_b)
        assert np.allclose(s_a, s_b, rtol=1e-4, atol=1e-5)

    def test_packed_matches_compressed(self):
        from remex import PackedVectors
        X = shifted_corpus(n=700, d=64, seed=2)
        pq = Quantizer(d=X.shape[1], bits=3, seed=42, mean=remex.corpus_mean(X))
        c = pq.encode(X)
        i_c, s_c = pq.search_adc(c, X[3], k=10)
        i_p, s_p = pq.search_adc(PackedVectors.from_compressed(c), X[3], k=10)
        assert np.array_equal(i_c, i_p)
        assert np.allclose(s_c, s_p, rtol=1e-4, atol=1e-5)

    def test_batch_matches_single(self):
        X = shifted_corpus(n=600, d=64, seed=4)
        pq = Quantizer(d=X.shape[1], bits=4, seed=42, mean=remex.corpus_mean(X))
        c = pq.encode(X)
        idx_b, _ = pq.search_batch(c, X[:3], k=5)
        for row, q in zip(idx_b, X[:3]):
            assert np.array_equal(row, pq.search(c, q, k=5)[0])

    def test_scores_match_the_score_vectors(self):
        X = shifted_corpus(n=400, d=64, seed=6)
        pq = Quantizer(d=X.shape[1], bits=4, seed=42, mean=remex.corpus_mean(X))
        c = pq.encode(X)
        q = X[9]
        idx, scores = pq.search(c, q, k=10)
        assert np.allclose(scores, score_ref(pq, c, q)[idx], rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("bits", [1, 2, 4])
    def test_scores_are_the_stated_estimator(self, bits):
        """q.mu + mu.r + ||r|| (q-mu).u_hat / kappa, computed from scratch."""
        X = shifted_corpus(n=300, d=64, seed=8)
        mu = remex.corpus_mean(X)
        pq = Quantizer(d=X.shape[1], bits=bits, seed=42, mean=mu)
        c = pq.encode(X)
        q = X[5] + 0.1
        r = X.astype(np.float64) - mu
        v = pq._unrotate_rows(pq.centroids[c.indices]).astype(np.float64)
        u_hat = v / np.linalg.norm(v, axis=1, keepdims=True)
        want = (q @ mu + r @ mu + np.linalg.norm(r, axis=1)
                * (u_hat @ (q - mu)) / pq._cosine_factor(None))
        idx, scores = pq.search(c, q, k=X.shape[0])
        assert np.allclose(scores, want[idx], rtol=1e-4, atol=1e-3)

    @pytest.mark.parametrize("bits", [1, 2])
    def test_estimator_is_unbiased(self, bits):
        """Mean signed error across all pairs is small against its spread.

        Which of ``/ kappa`` (unbiased, RaBitQ) and ``* kappa`` (shrunk) ranks
        better is corpus-dependent: on SPECTER2 ``/ kappa`` wins at 1 bit by
        0.15 recall@10, on isotropic synthetic data ``* kappa`` wins by about
        0.04. Search follows the real corpus; ``bench/centered_eval.py`` has
        the numbers. This test pins the unbiasedness, not the ranking.
        """
        X = shifted_corpus(n=1500, d=96, shift=3.0, seed=3)
        queries, corpus = X[:60], X[60:]
        pq = Quantizer(d=96, bits=bits, seed=42, mean=remex.corpus_mean(corpus))
        c = pq.encode(corpus)
        err = np.stack([score_ref(pq, c, q) for q in queries]) - queries @ corpus.T
        assert abs(err.mean()) < 0.05 * err.std()

    def test_centred_search_beats_uncentred_on_a_shifted_corpus(self):
        X = shifted_corpus(n=1500, d=96, shift=3.0, seed=3)
        queries, corpus = X[:60], X[60:]
        truth = topk(corpus, queries, 10)
        got = {}
        for mean in (None, remex.corpus_mean(corpus)):
            pq = Quantizer(d=96, bits=1, seed=42, mean=mean)
            got[mean is None] = recall(pq.search_batch(pq.encode(corpus), queries, k=10)[0], truth, 10)
        assert got[False] > got[True] + 0.05


class TestRecall:
    @pytest.mark.parametrize("bits,margin", [(1, 0.03), (2, 0.03), (4, 0.02)])
    def test_centering_helps_a_shifted_corpus(self, bits, margin):
        X = shifted_corpus(n=1500, d=96, seed=3)
        queries, corpus = X[:60], X[60:]
        truth = topk(corpus, queries, 10)

        # The margins were calibrated on this draw under haar, the default
        # before 1.0. They sit near the noise floor at 4 bits for either
        # rotation: over data seeds 3-8 the 4-bit gain ranged 0.017-0.065
        # under haar and 0.012-0.040 under rht (mean 0.039 vs 0.026).
        def r10(mean):
            pq = Quantizer(d=corpus.shape[1], bits=bits, seed=42, mean=mean,
                           rotation="haar")
            return recall(topk(pq.decode(pq.encode(corpus)), queries, 10), truth, 10)

        assert r10(remex.corpus_mean(corpus)) >= r10(None) + margin


class TestSerialization:
    def test_npz_roundtrip(self, tmp_path):
        from remex import CompressedVectors
        X = shifted_corpus(n=300, d=64, seed=7)
        mu = remex.corpus_mean(X)
        pq = Quantizer(d=64, bits=4, seed=42, mean=mu)
        c = pq.encode(X)
        p = tmp_path / "c.npz"
        c.save(str(p))
        back = CompressedVectors.load(str(p))
        assert np.allclose(back.mean, mu)
        assert np.allclose(pq.decode(back), pq.decode(c))

    # Encodes the shifted corpus uncentered on purpose, which is exactly what
    # AnisotropyWarning exists to flag.
    @pytest.mark.filterwarnings("ignore::remex.AnisotropyWarning")
    def test_npz_plain_roundtrip_has_no_mean(self, tmp_path):
        from remex import CompressedVectors
        X = shifted_corpus(n=300, d=64, seed=8)
        c = Quantizer(d=64, bits=4, seed=42).encode(X)
        p = tmp_path / "p.npz"
        c.save(str(p))
        assert CompressedVectors.load(str(p)).mean is None

    def test_pq_refuses_centered_container(self, tmp_path):
        from remex.pq_format import save_pq
        X = shifted_corpus(n=200, d=64, seed=9)
        c = Quantizer(d=64, bits=4, seed=42, mean=remex.corpus_mean(X)).encode(X)
        with pytest.raises(ValueError, match="centered"):
            save_pq(str(tmp_path / "c.pq"), c)


class TestIvfAndGpuAgree:
    """The paths that reimplement scoring have to carry the centre too.

    Both read ``_effective_norms`` already; neither knew about the constant a
    centre adds, and nothing else in the suite would have noticed, because
    every other cross-path test uses an uncentred quantizer.
    """

    def test_ivf_full_nprobe_matches_search_adc(self):
        from remex import IVFCoarseIndex
        X = shifted_corpus(n=800, d=64, seed=11)
        pq = Quantizer(d=X.shape[1], bits=4, seed=42, mean=remex.corpus_mean(X))
        c = pq.encode(X)
        ivf = IVFCoarseIndex(pq, c, n_bits=4)
        q = X[17]
        i_ivf, s_ivf = ivf.search_coarse(q, k=10, nprobe=2 ** 4)
        i_adc, s_adc = pq.search_adc(c, q, k=10)
        assert np.array_equal(i_ivf, i_adc)
        assert np.allclose(s_ivf, s_adc, rtol=1e-4, atol=1e-4)

    def test_gpu_matches_quantizer(self):
        from remex.gpu import GPUSearcher
        X = shifted_corpus(n=800, d=64, seed=12)
        pq = Quantizer(d=X.shape[1], bits=4, seed=42, mean=remex.corpus_mean(X))
        c = pq.encode(X)
        g = GPUSearcher(pq, c, backend="numpy")
        q = X[23]
        i_g, s_g = g.search(q, k=10)
        i_q, s_q = pq.search(c, q, k=10)
        assert np.array_equal(i_g, i_q)
        assert np.allclose(s_g, s_q, rtol=1e-4, atol=1e-4)


class TestMeanDots:
    def test_columns_are_exact(self):
        X = shifted_corpus(seed=4)
        mu = remex.corpus_mean(X)
        c = Quantizer(d=X.shape[1], bits=2, seed=42, mean=mu).encode(X)
        r = X.astype(np.float64) - mu
        assert np.allclose(c.norms, np.linalg.norm(r, axis=1), rtol=1e-5)
        assert np.allclose(c.mean_dots, r @ mu, rtol=1e-4, atol=1e-3)

    def test_plain_container_has_none(self):
        X = shifted_corpus()
        assert Quantizer(d=X.shape[1], bits=4, seed=42).encode(X).mean_dots is None

    def test_nbytes_counts_the_column(self):
        X = shifted_corpus(n=100, d=64)
        c = Quantizer(d=64, bits=2, seed=42, mean=remex.corpus_mean(X)).encode(X)
        assert c.nbytes == 100 * (64 * 2 // 8) + 100 * 4 + 100 * 4

    def test_subset_and_packed_carry_it(self):
        from remex import PackedVectors
        X = shifted_corpus()
        c = Quantizer(d=X.shape[1], bits=4, seed=42, mean=remex.corpus_mean(X)).encode(X)
        idx = np.array([5, 1, 9])
        assert np.array_equal(c.subset(idx).mean_dots, c.mean_dots[idx])
        p = PackedVectors.from_compressed(c)
        assert np.array_equal(p.mean_dots, c.mean_dots)
        assert np.array_equal(p.subset(idx).mean_dots, c.mean_dots[idx])
        assert np.array_equal(p.at_precision(2).mean_dots, c.mean_dots)
        assert np.array_equal(p.to_compressed().mean_dots, c.mean_dots)

    def test_npz_roundtrip(self, tmp_path):
        from remex import CompressedVectors, PackedVectors
        X = shifted_corpus()
        pq = Quantizer(d=X.shape[1], bits=4, seed=42, mean=remex.corpus_mean(X))
        c = pq.encode(X)
        c.save(tmp_path / "c.npz")
        back = CompressedVectors.load(tmp_path / "c.npz")
        assert np.array_equal(back.mean_dots, c.mean_dots)
        PackedVectors.from_compressed(c).save(tmp_path / "p.npz")
        pback = PackedVectors.load(tmp_path / "p.npz")
        assert np.array_equal(pback.mean_dots, c.mean_dots)
        q = X[3]
        assert np.allclose(pq.search(back, q, 5)[1], pq.search(c, q, 5)[1])

    def test_arrow_roundtrip_keeps_mean_and_mean_dots(self, tmp_path):
        pytest.importorskip("pyarrow")
        from remex import CompressedVectors
        X = shifted_corpus()
        pq = Quantizer(d=X.shape[1], bits=4, seed=42, mean=remex.corpus_mean(X))
        c = pq.encode(X)
        c.save_arrow(str(tmp_path / "c.arrow"), seed=42)
        back = CompressedVectors.load_arrow(str(tmp_path / "c.arrow"))
        assert np.array_equal(back.mean, c.mean)
        assert np.array_equal(back.mean_dots, c.mean_dots)
        assert np.allclose(pq.decode(back), pq.decode(c))


class TestCosineFactor:
    @pytest.mark.parametrize("bits,want", [(1, 0.798), (2, 0.940), (3, 0.983), (4, 0.995)])
    def test_direct_widths(self, bits, want):
        pq = Quantizer(d=768, bits=bits, seed=42)
        assert abs(pq._cosine_factor(None) - want) < 0.002

    def test_matches_measured_cosine(self):
        """kappa is the mean cosine between u and u_hat, at every level."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((2000, 256)).astype(np.float32)
        pq = Quantizer(d=256, bits=8, seed=42)
        c = pq.encode(X)
        U = X / np.linalg.norm(X, axis=1, keepdims=True)
        for p in (1, 2, 4, 8):
            V = pq._unrotate_rows(pq._resolve_centroids(c, p)[pq._resolve_indices(c, p)])
            cos = np.sum(V * U, axis=1) / np.linalg.norm(V, axis=1)
            assert abs(cos.mean() - pq._cosine_factor(p)) < 0.003
