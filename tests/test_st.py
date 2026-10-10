"""remex.st: RemexIndex and semantic_search_remex over a sentence-transformers model."""
import warnings

import numpy as np
import pytest

from remex import (
    AnisotropyWarning, CompressedVectors, Quantizer, RemexIndex,
    semantic_search_remex,
)
from _fake_st import FakeModel

DOCS = [f"document number {i} about topic {i % 7}" for i in range(300)]
QUERIES = [f"question {j}" for j in range(12)]


def _unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


@pytest.fixture
def model():
    return FakeModel(dim=64)


@pytest.fixture
def index(model):
    idx = RemexIndex(model, bits=4)
    idx.add(DOCS)
    return idx


class TestShape:
    def test_single_query_returns_one_list(self, index):
        hits = index.search(QUERIES[0], top_k=5)
        assert len(hits) == 5
        assert set(hits[0]) == {"corpus_id", "score"}
        assert isinstance(hits[0]["corpus_id"], int)
        assert isinstance(hits[0]["score"], float)
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_query_list_returns_list_per_query(self, index):
        hits = index.search(QUERIES, top_k=3)
        assert len(hits) == len(QUERIES)
        assert all(len(h) == 3 for h in hits)

    def test_list_of_one_string_is_not_single(self, index):
        hits = index.search(QUERIES[:1], top_k=3)
        assert len(hits) == 1 and len(hits[0]) == 3

    def test_top_k_capped_at_corpus_size(self, model):
        idx = RemexIndex(model, bits=4)
        idx.add(DOCS[:4])
        assert len(idx.search("q", top_k=10)) == 4

    def test_add_returns_ids_in_insertion_order(self, model):
        idx = RemexIndex(model, bits=4)
        assert list(idx.add(DOCS[:3])) == [0, 1, 2]
        assert list(idx.add(DOCS[3:5])) == [3, 4]
        assert len(idx) == 5


class TestMatchesQuantizer:
    """The wrapper must rank exactly as the quantizer it hides."""

    def test_same_ids_and_scores_as_quantizer_search(self, model, index):
        D = _unit(model.encode(DOCS))
        pq = Quantizer(d=64, bits=4, seed=42)
        codes = pq.encode(D)
        for q in QUERIES:
            qv = _unit(model.encode([q]))[0]
            ids, scores = pq.search(codes, qv, k=10)
            hits = index.search(q, top_k=10)
            assert [h["corpus_id"] for h in hits] == list(ids)
            np.testing.assert_allclose([h["score"] for h in hits], scores, rtol=1e-5)

    def test_incremental_add_equals_one_add(self, model, index):
        idx = RemexIndex(model, bits=4)
        for lo in range(0, len(DOCS), 70):
            idx.add(DOCS[lo:lo + 70])
        np.testing.assert_array_equal(idx.compressed.indices, index.compressed.indices)
        np.testing.assert_array_equal(idx.compressed.norms, index.compressed.norms)
        assert idx.search(QUERIES, top_k=5) == index.search(QUERIES, top_k=5)

    def test_low_memory_ranks_identically(self, model, index):
        lm = RemexIndex(model, bits=4, low_memory=True)
        lm.add(DOCS)
        a = lm.search(QUERIES, top_k=10)
        b = index.search(QUERIES, top_k=10)
        assert [[h["corpus_id"] for h in r] for r in a] == \
               [[h["corpus_id"] for h in r] for r in b]

    def test_8bit_recall_against_exact(self, model):
        D = _unit(model.encode(DOCS))
        Q = _unit(model.encode(QUERIES))
        exact = np.argsort(-(Q @ D.T), axis=1)[:, :10]
        idx = RemexIndex(model, bits=8)
        idx.add(DOCS)
        got = idx.search(QUERIES, top_k=10)
        recall = np.mean([
            len(set(exact[i]) & {h["corpus_id"] for h in got[i]}) / 10
            for i in range(len(QUERIES))
        ])
        assert recall >= 0.9


class TestModelSurface:
    def test_query_and_document_encoders_are_used(self, model):
        idx = RemexIndex(model, bits=4)
        idx.add(DOCS[:10])
        idx.search("q")
        kinds = [c[0] for c in model.calls]
        assert kinds.index("document") < kinds.index("query")

    def test_cosine_asks_model_to_normalize(self, model):
        RemexIndex(model, bits=4).add(DOCS[:3])
        assert model.calls[-1][2] is True

    def test_encode_kwargs_pass_through(self, model):
        RemexIndex(model, bits=4).add(DOCS[:3], batch_size=7)
        assert model.calls[-1][3] == {"batch_size": 7}

    def test_dim_comes_from_model(self):
        assert RemexIndex(FakeModel(dim=96)).dim == 96

    def test_similarity_comes_from_model(self):
        assert RemexIndex(FakeModel(similarity_fn_name="dot")).similarity == "dot"

    def test_distance_similarities_are_refused(self):
        with pytest.raises(ValueError, match="inner product"):
            RemexIndex(FakeModel(similarity_fn_name="euclidean"))

    def test_dot_keeps_lengths(self):
        rng = np.random.default_rng(0)
        D = _unit(rng.standard_normal((50, 32)).astype(np.float32))
        D[7] *= 10  # same direction class as the rest, ten times longer
        idx = RemexIndex(bits=8, similarity="dot")
        idx.add(D)
        q = D[7] / 10
        assert idx.search(q, top_k=1)[0]["corpus_id"] == 7
        assert idx.search(q, top_k=1)[0]["score"] == pytest.approx(10.0, rel=0.05)

    def test_bad_bits_raise_at_construction(self, model):
        with pytest.raises(ValueError, match="bits"):
            RemexIndex(model, bits=5)


class TestEmbeddingsOnly:
    def test_add_and_search_arrays(self):
        rng = np.random.default_rng(1)
        D = rng.standard_normal((100, 48)).astype(np.float32)
        idx = RemexIndex(bits=4)
        idx.add(D)
        assert idx.dim == 48
        assert idx.search(D[3], top_k=1)[0]["corpus_id"] == 3

    def test_text_without_model_raises(self):
        idx = RemexIndex(bits=4, dim=8)
        with pytest.raises(ValueError, match="no model"):
            idx.add(["text"])

    def test_dim_mismatch_raises(self, index):
        with pytest.raises(ValueError, match="dim=64"):
            index.search(np.ones(10, np.float32))

    def test_empty_index_raises(self, model):
        with pytest.raises(ValueError, match="empty"):
            RemexIndex(model).search("q")


class TestCentering:
    def _shared_direction(self):
        rng = np.random.default_rng(2)
        return (rng.standard_normal((400, 64)) * 0.2 + 3.0).astype(np.float32)

    def test_warning_points_at_center_flag(self):
        idx = RemexIndex(bits=4, similarity="dot")
        with pytest.warns(AnisotropyWarning, match=r"center=True") as rec:
            idx.add(self._shared_direction())
        assert all("Quantizer(" not in str(w.message) for w in rec)

    def test_center_uses_first_batch_mean_and_is_quiet(self):
        X = self._shared_direction()
        idx = RemexIndex(bits=4, similarity="dot", center=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", AnisotropyWarning)
            idx.add(X[:300])
            idx.add(X[300:])
        np.testing.assert_allclose(
            idx.quantizer.mean, X[:300].astype(np.float64).mean(axis=0), rtol=1e-5
        )
        assert len(idx) == 400


class TestPersistence:
    def test_round_trip(self, model, index, tmp_path):
        path = index.save(tmp_path / "idx")
        assert path.endswith(".npz")
        again = RemexIndex.load(path, model)
        assert again.search(QUERIES, top_k=10) == index.search(QUERIES, top_k=10)

    def test_seed_survives(self, model, tmp_path):
        idx = RemexIndex(model, bits=4, seed=7)
        idx.add(DOCS)
        again = RemexIndex.load(idx.save(tmp_path / "s7"), model)
        assert again.seed == 7
        assert again.search(QUERIES, top_k=10) == idx.search(QUERIES, top_k=10)

    def test_centered_round_trip(self, tmp_path):
        rng = np.random.default_rng(3)
        X = (rng.standard_normal((200, 32)) + 2).astype(np.float32)
        idx = RemexIndex(bits=4, similarity="dot", center=True)
        idx.add(X)
        again = RemexIndex.load(idx.save(tmp_path / "c"))
        assert again.center and again.similarity == "dot"
        assert again.search(X[:5], top_k=5) == idx.search(X[:5], top_k=5)

    def test_file_is_a_compressed_vectors_container(self, index, tmp_path):
        codes = CompressedVectors.load(index.save(tmp_path / "idx"))
        np.testing.assert_array_equal(codes.indices, index.compressed.indices)

    def test_plain_container_is_refused(self, index, tmp_path):
        path = str(tmp_path / "plain.npz")
        index.compressed.save(path)
        with pytest.raises(ValueError, match="seed"):
            RemexIndex.load(path)

    def test_model_dimension_checked(self, index, tmp_path):
        path = index.save(tmp_path / "idx")
        with pytest.raises(ValueError, match="dim=64"):
            RemexIndex.load(path, FakeModel(dim=32))

    def test_empty_index_refuses_to_save(self, model, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            RemexIndex(model).save(tmp_path / "x")


class TestSemanticSearchRemex:
    def test_faiss_shaped_return(self):
        rng = np.random.default_rng(4)
        D = rng.standard_normal((200, 32)).astype(np.float32)
        results, seconds = semantic_search_remex(D[:3], D, top_k=4)
        assert len(results) == 3 and all(len(r) == 4 for r in results)
        assert [r[0]["corpus_id"] for r in results] == [0, 1, 2]
        assert seconds >= 0.0

    def test_index_reuse(self):
        rng = np.random.default_rng(5)
        D = rng.standard_normal((200, 32)).astype(np.float32)
        first, _, idx = semantic_search_remex(D[:2], D, output_index=True)
        again, _ = semantic_search_remex(D[:2], corpus_index=idx)
        assert first == again

    def test_needs_a_corpus(self):
        with pytest.raises(ValueError, match="corpus"):
            semantic_search_remex(np.ones((1, 8), np.float32))


# --- late interaction (MultiVectorEncoder, maxsim) -------------------------

from _fake_st import FakeMultiVectorModel  # noqa: E402
import remex.st as remex_st  # noqa: E402

_WORDS = [f"w{i}" for i in range(400)]


def _mv_docs(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(_WORDS, rng.integers(3, 25))) for _ in range(n)]


MV_DOCS = _mv_docs()
MV_QUERIES = [" ".join(d.split()[:4]) for d in MV_DOCS[:10]]


def _maxsim(q, docs):
    return np.array([(q @ d.T).max(axis=1).sum() for d in docs])


@pytest.fixture
def mv_model():
    return FakeMultiVectorModel(dim=32)


@pytest.fixture
def mv_index(mv_model):
    idx = RemexIndex(mv_model, bits=4)
    idx.add(MV_DOCS)
    return idx


class TestLateInteraction:
    def test_similarity_and_length_come_from_the_model(self, mv_index):
        assert mv_index.multi_vector
        assert len(mv_index) == len(MV_DOCS)
        assert mv_index.offsets[-1] == mv_index.compressed.n

    def test_scores_are_maxsim_over_decoded_tokens(self, mv_model, mv_index):
        decoded = mv_index.quantizer.decode(mv_index.compressed)
        off = mv_index.offsets
        docs = [decoded[off[i]:off[i + 1]] for i in range(len(MV_DOCS))]
        for text in MV_QUERIES:
            q = mv_model.encode([text])[0]
            want = _maxsim(q, docs)
            hits = mv_index.search(text, top_k=5)
            np.testing.assert_allclose(
                [h["score"] for h in hits], np.sort(want)[::-1][:5], rtol=1e-5)
            assert [h["corpus_id"] for h in hits][0] == int(np.argmax(want))

    def test_8bit_recall_against_float_maxsim(self, mv_model):
        docs = mv_model.encode(MV_DOCS)
        idx = RemexIndex(mv_model, bits=8)
        idx.add(MV_DOCS)
        recall = []
        for text in MV_QUERIES:
            exact = set(np.argsort(-_maxsim(mv_model.encode([text])[0], docs))[:10])
            got = {h["corpus_id"] for h in idx.search(text, top_k=10)}
            recall.append(len(exact & got) / 10)
        assert np.mean(recall) >= 0.9

    def test_query_matrix_and_list_of_matrices(self, mv_model, mv_index):
        q = mv_model.encode(MV_QUERIES[:2])
        assert len(mv_index.search(q[0], top_k=3)) == 3            # one query
        assert len(mv_index.search(q, top_k=3)) == 2               # two queries
        assert mv_index.search(q[0], top_k=3) == mv_index.search(MV_QUERIES[0], top_k=3)

    def test_blocking_does_not_change_results(self, mv_index, monkeypatch):
        whole = mv_index.search(MV_QUERIES, top_k=10)
        monkeypatch.setattr(remex_st, "_SCORE_BLOCK_FLOATS", 64)  # a few docs per block
        blocked = mv_index.search(MV_QUERIES, top_k=10)
        # block shape changes BLAS summation order, so scores agree to rounding
        assert [[h["corpus_id"] for h in r] for r in blocked] == \
               [[h["corpus_id"] for h in r] for r in whole]
        np.testing.assert_allclose([[h["score"] for h in r] for r in blocked],
                                   [[h["score"] for h in r] for r in whole], rtol=1e-5)

    def test_low_memory_identical(self, mv_model, mv_index):
        lm = RemexIndex(mv_model, bits=4, low_memory=True)
        lm.add(MV_DOCS)
        assert lm.search(MV_QUERIES, top_k=10) == mv_index.search(MV_QUERIES, top_k=10)
        assert lm._decoded is None

    def test_incremental_add(self, mv_model, mv_index):
        idx = RemexIndex(mv_model, bits=4)
        assert list(idx.add(MV_DOCS[:50])) == list(range(50))
        assert list(idx.add(MV_DOCS[50:])) == list(range(50, len(MV_DOCS)))
        np.testing.assert_array_equal(idx.offsets, mv_index.offsets)
        assert idx.search(MV_QUERIES, top_k=10) == mv_index.search(MV_QUERIES, top_k=10)

    def test_add_invalidates_decoded_cache(self, mv_model):
        idx = RemexIndex(mv_model, bits=4)
        idx.add(MV_DOCS[:50])
        idx.search(MV_QUERIES[0])
        idx.add(MV_DOCS[50:])
        assert idx.search(MV_QUERIES[0], top_k=1)[0]["corpus_id"] == 0

    def test_meanmaxsim_divides_by_query_tokens(self, mv_model, mv_index):
        idx = RemexIndex(mv_model, bits=4, similarity="meanmaxsim")
        idx.add(MV_DOCS)
        a = mv_index.search(MV_QUERIES[0], top_k=5)
        b = idx.search(MV_QUERIES[0], top_k=5)
        n_tok = len(MV_QUERIES[0].split())
        assert [h["corpus_id"] for h in a] == [h["corpus_id"] for h in b]
        np.testing.assert_allclose([h["score"] / n_tok for h in a],
                                   [h["score"] for h in b], rtol=1e-5)

    def test_token_matrices_without_model(self, mv_model):
        idx = RemexIndex(bits=4, similarity="maxsim")
        idx.add(mv_model.encode(MV_DOCS))
        assert idx.dim == 32
        assert idx.search(mv_model.encode([MV_DOCS[7]])[0], top_k=1)[0]["corpus_id"] == 7

    def test_empty_document_refused(self):
        idx = RemexIndex(bits=4, similarity="maxsim")
        with pytest.raises(ValueError, match="non-empty"):
            idx.add([np.ones((3, 8), np.float32), np.ones((0, 8), np.float32)])

    def test_shared_direction_warns_toward_center(self):
        model = FakeMultiVectorModel(dim=32, shared=3.0)
        with pytest.warns(AnisotropyWarning, match=r"center=True"):
            RemexIndex(model, bits=2).add(MV_DOCS)
        with warnings.catch_warnings():
            warnings.simplefilter("error", AnisotropyWarning)
            RemexIndex(model, bits=2, center=True).add(MV_DOCS)

    def test_centering_helps_on_shared_direction(self):
        model = FakeMultiVectorModel(dim=32, shared=3.0)
        docs = model.encode(MV_DOCS)
        exact = [set(np.argsort(-_maxsim(model.encode([q])[0], docs))[:10])
                 for q in MV_QUERIES]

        def recall(center):
            idx = RemexIndex(model, bits=1, center=center)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AnisotropyWarning)
                idx.add(MV_DOCS)
            got = idx.search(MV_QUERIES, top_k=10)
            return np.mean([len(e & {h["corpus_id"] for h in g}) / 10
                            for e, g in zip(exact, got)])

        assert recall(True) > recall(False)

    def test_round_trip(self, mv_model, mv_index, tmp_path):
        again = RemexIndex.load(mv_index.save(tmp_path / "mv"), mv_model)
        assert again.multi_vector
        np.testing.assert_array_equal(again.offsets, mv_index.offsets)
        assert again.search(MV_QUERIES, top_k=10) == mv_index.search(MV_QUERIES, top_k=10)
