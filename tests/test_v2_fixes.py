"""Regressions found by the 2.0 test fan-out, one test per finding."""
import numpy as np
import pytest

import remex
from remex import CompressedVectors, PackedVectors, Quantizer


def shifted(n=400, d=64, shift=3.0, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(d) * shift
    return (mu + rng.standard_normal((n, d))).astype(np.float32)


def centred(X, bits=2):
    pq = Quantizer(d=X.shape[1], bits=bits, seed=42, mean=remex.corpus_mean(X))
    return pq, pq.encode(X)


def test_decode_on_an_extreme_shift_beats_the_uncorrected_direction():
    """1.x solved a quadratic for the decode length and could take a root on
    the far side of the mean (decode MSE 47 against 26 here). 2.0 decodes
    ``w mu + kappa ||r|| u_hat`` and has no root to choose."""
    X = shifted(n=2000, shift=10.0, seed=1)
    pq, c = centred(X, bits=1)
    v = pq._unrotate_rows(pq.centroids[c.indices])
    plain = c.mean + c.norms[:, None] * v / np.linalg.norm(v, axis=1, keepdims=True)
    err = np.sum((pq.decode(c) - X) ** 2, axis=1)
    assert np.all(np.isfinite(err))
    assert err.mean() <= np.sum((plain - X) ** 2, axis=1).mean()


@pytest.mark.parametrize("d,bits", [(2, 1), (3, 1), (7, 1), (16, 2), (64, 1)])
def test_kappa_matches_the_measured_cosine_at_small_d(d, bits):
    rng = np.random.default_rng(0)
    U = rng.standard_normal((40000, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    pq = Quantizer(d=d, bits=bits, seed=42, rotation="haar")
    c = pq.encode(U.astype(np.float32))
    V = pq._unrotate_rows(pq.centroids[c.indices]).astype(np.float64)
    cos = np.sum(V * U, axis=1) / np.linalg.norm(V, axis=1)
    assert abs(cos.mean() - pq._cosine_factor(None)) < 0.012


def test_a_wrong_mean_is_refused_after_the_caches_are_warm():
    X = shifted()
    mu = remex.corpus_mean(X)
    right = Quantizer(d=64, bits=2, seed=42, mean=mu)
    wrong = Quantizer(d=64, bits=2, seed=42, mean=mu + 1.0)
    c = right.encode(X)
    right.search(c, X[0], k=5)
    right.search_batch(c, X[:2], k=5)
    for call in (lambda: wrong.search(c, X[0], k=5),
                 lambda: wrong.search_batch(c, X[:2], k=5),
                 lambda: wrong.search_adc(c, X[0], k=5)):
        with pytest.raises(ValueError, match="mean mismatch"):
            call()


def test_float64_input_keeps_the_residual_exact():
    rng = np.random.default_rng(2)
    X = 1000.0 + rng.standard_normal((50, 16)) * 1e-3
    mu = X.mean(axis=0)
    pq = Quantizer(d=16, bits=2, seed=42, mean=mu)
    c = pq.encode(X)
    r = X - pq.mean.astype(np.float64)
    assert np.allclose(c.norms, np.linalg.norm(r, axis=1), rtol=1e-4)


def test_boolean_mask_subset_agrees_across_containers():
    X = shifted()
    pq, c = centred(X)
    mask = np.zeros(len(X), bool)
    mask[::3] = True
    p = PackedVectors.from_compressed(c).subset(mask)
    assert p.n == mask.sum() == c.subset(mask).n
    assert np.array_equal(p.mean_dots, c.mean_dots[mask])
    assert np.array_equal(pq.search_adc(p, X[0], k=5)[0], pq.search_adc(c.subset(mask), X[0], k=5)[0])


def test_arrow_zero_rows_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    X = shifted()
    _, c = centred(X)
    empty = c.subset(np.array([], dtype=int))
    empty.save_arrow(str(tmp_path / "e.arrow"))
    back = CompressedVectors.load_arrow(str(tmp_path / "e.arrow"))
    assert back.n == 0 and back.mean_dots.shape == (0,)


def test_mean_dots_is_validated():
    X = shifted(n=20)
    pq, c = centred(X)
    with pytest.raises(ValueError, match="rows"):
        CompressedVectors(c.indices, c.norms, c.d, c.bits, c.rotation, c.mean, c.mean_dots[:-1])
    with pytest.raises(ValueError, match="only meaningful with a mean"):
        CompressedVectors(c.indices, c.norms, c.d, c.bits, c.rotation, None, c.mean_dots)
    p = PackedVectors.from_compressed(c)
    with pytest.raises(ValueError, match="mean_dots"):
        PackedVectors.from_rows(list(p._packed), c.norms, c.d, c.bits, c.rotation, mean=c.mean)
    ok = PackedVectors.from_rows(list(p._packed), c.norms, c.d, c.bits, c.rotation,
                                 mean=c.mean, mean_dots=c.mean_dots)
    assert np.allclose(pq.search_adc(ok, X[0], k=3)[1], pq.search_adc(c, X[0], k=3)[1])


def test_score_vectors_accepts_packed():
    X = shifted()
    pq, c = centred(X)
    Y, b = pq._score_vectors(PackedVectors.from_compressed(c), precision=1)
    Y2, b2 = pq._score_vectors(c, precision=1)
    assert np.allclose(Y, Y2) and np.allclose(b, b2)


def test_gpu_refuses_a_scalar_container_for_a_centred_quantizer():
    from remex.gpu import GPUSearcher
    X = shifted()
    pq, _ = centred(X)
    scalar = Quantizer(d=64, bits=2, seed=42, normalize=False).encode(X)
    with pytest.raises(ValueError):
        GPUSearcher(pq, scalar, backend="numpy")


def test_gpu_search_batch_accepts_one_query():
    from remex.gpu import GPUSearcher
    X = shifted()
    pq, c = centred(X)
    ids, _ = GPUSearcher(pq, c, backend="numpy").search_batch(X[0], k=3)
    assert ids.shape == (1, 3)
    assert np.array_equal(ids, pq.search_batch(c, X[0], k=3)[0])


def test_torch_backend_takes_a_reversed_query():
    pytest.importorskip("torch")
    from remex.gpu import GPUSearcher
    X = shifted()
    pq, c = centred(X)
    q = X[0][::-1]
    try:
        searcher = GPUSearcher(pq, c, backend="torch")
    except ImportError:
        pytest.skip("torch backend needs a CUDA device")
    ids, _ = searcher.search(q, k=5)
    assert np.array_equal(ids, pq.search(c, q, k=5)[0])


def test_remex_index_load_refuses_a_pre_2_0_centred_file(tmp_path):
    from remex.st import RemexIndex
    X = shifted()
    idx = RemexIndex(None, bits=2, center=True, similarity="dot", dim=64)
    idx.add(X)
    idx.save(str(tmp_path / "i.npz"))
    data = dict(np.load(tmp_path / "i.npz"))
    del data["mean_dots"]
    np.savez(tmp_path / "old.npz", **data)
    with pytest.raises(ValueError, match="remex < 2.0"):
        RemexIndex.load(str(tmp_path / "old.npz"))


def test_kappa_is_one_at_d_1():
    X = np.array([[3.0], [2.5], [4.0], [-1.0]], np.float32)
    pq, c = centred(X, bits=2)
    assert pq._cosine_factor(None) == 1.0
    assert np.all(np.isfinite(pq.search(c, X[0], k=4)[1]))


def test_a_different_bit_width_is_refused():
    X = shifted()
    c = Quantizer(d=64, bits=2, seed=42).encode(X)
    q4 = Quantizer(d=64, bits=4, seed=42)
    for call in (lambda: q4.decode(c), lambda: q4.search(c, X[0], k=3),
                 lambda: q4.search_adc(c, X[0], k=3)):
        with pytest.raises(ValueError, match="bits mismatch"):
            call()


def test_subset_edge_selections():
    X = shifted(n=50)
    _, c = centred(X)
    p = PackedVectors.from_compressed(c)
    for cont in (c, p):
        assert cont.subset([]).n == 0
        with pytest.raises(IndexError):
            cont.subset(np.ones(7, bool))


def test_ivf_empty_cell_still_runs_the_guards():
    from remex import IVFCoarseIndex
    X = shifted(n=40)
    mu = remex.corpus_mean(X)
    c = Quantizer(d=64, bits=2, seed=42, mean=mu).encode(X)
    wrong = Quantizer(d=64, bits=2, seed=42, mean=mu + 1.0)
    ivf = IVFCoarseIndex(wrong, c, n_bits=12)
    with pytest.raises(ValueError, match="mean mismatch"):
        for q in X:
            ivf.search_coarse(q, k=5, nprobe=1)


def test_warm_scalar_search_batch_runs_the_guards():
    X = shifted()
    s = Quantizer(d=64, bits=2, seed=42, normalize=False, rotation="haar")
    c = s.encode(X)
    s.search_batch(c, X[:2], k=3)
    other = Quantizer(d=64, bits=2, seed=42, normalize=False, rotation="rht")
    with pytest.raises(ValueError, match="rotation mismatch"):
        other.search_batch(c, X[:2], k=3)
