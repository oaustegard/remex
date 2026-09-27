"""Centered mode: encode the offset from a caller-declared corpus mean.

remex stays training-free by never measuring the mean itself. The caller
declares it, exactly as it declares ``scale`` in scalar mode, and
``remex.corpus_mean`` is the blessed way to compute one.

The reconstruction is ``mu + m * u_hat`` with ``m`` solved at encode time so
the reconstruction's length equals the original vector's. That keeps the
correction in the norms column the container already has, so no per-vector
byte is added. Subtracting a mean WITHOUT restoring the full length loses
recall at every bit width measured, which is why the two are one feature.
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
    def test_length_is_restored(self):
        """The whole point: ``||mu + m*u|| == ||x||``, not ``||m*u|| == ||r||``."""
        X = shifted_corpus()
        pq = Quantizer(d=X.shape[1], bits=3, seed=42, mean=remex.corpus_mean(X))
        recon = pq.decode(pq.encode(X))
        assert np.allclose(np.linalg.norm(recon, axis=1),
                           np.linalg.norm(X, axis=1), rtol=1e-3)

    def test_extreme_shift_falls_back_on_a_small_minority(self):
        """No positive root exists when the mean dwarfs the spread. Those rows
        keep the residual's own length instead, and stay finite and positive."""
        X = shifted_corpus(shift=3.0)
        pq = Quantizer(d=X.shape[1], bits=3, seed=42, mean=remex.corpus_mean(X))
        c = pq.encode(X)
        assert np.all(np.isfinite(c.norms)) and np.all(c.norms > 0)
        rel = np.abs(np.linalg.norm(pq.decode(c), axis=1)
                     - np.linalg.norm(X, axis=1)) / np.linalg.norm(X, axis=1)
        assert (rel > 1e-3).sum() < 0.02 * len(X)

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

    def test_scores_track_the_reconstruction(self):
        X = shifted_corpus(n=400, d=64, seed=6)
        pq = Quantizer(d=X.shape[1], bits=4, seed=42, mean=remex.corpus_mean(X))
        c = pq.encode(X)
        q = X[9]
        idx, scores = pq.search(c, q, k=10)
        assert np.allclose(scores, pq.decode(c)[idx] @ q, rtol=1e-3, atol=1e-4)


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
