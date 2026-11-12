"""RemexIndex(token_pooling=...): sentence-transformers 6.3 token pooling."""
import sys
import types

import numpy as np
import pytest

from remex import RemexIndex
from _fake_st import FakeModel, FakeMultiVectorModel, FakeTokenPooling

DOCS = [" ".join(f"w{(i * 7 + j) % 50}" for j in range(5 + i % 9)) for i in range(60)]
QUERY = "w3 w10 w17"


@pytest.fixture
def mv_model():
    return FakeMultiVectorModel(dim=32)


@pytest.fixture
def fake_modules(monkeypatch):
    """Make ``sentence_transformers.multi_vector_encoder.modules`` importable
    with the stand-in pooling module, as sentence-transformers 6.3 provides it."""
    mods = types.ModuleType("sentence_transformers.multi_vector_encoder.modules")
    mods.FakeTokenPooling = FakeTokenPooling
    mods.HierarchicalTokenPooling = FakeTokenPooling
    pkg = types.ModuleType("sentence_transformers.multi_vector_encoder")
    pkg.modules = mods
    root = types.ModuleType("sentence_transformers")
    root.multi_vector_encoder = pkg
    monkeypatch.setitem(sys.modules, "sentence_transformers", root)
    monkeypatch.setitem(sys.modules, "sentence_transformers.multi_vector_encoder", pkg)
    monkeypatch.setitem(sys.modules, "sentence_transformers.multi_vector_encoder.modules", mods)


class TestPooledAdd:
    def test_stores_the_pooled_tokens(self, mv_model):
        pooling = FakeTokenPooling(pool_factor=2)
        idx = RemexIndex(mv_model, bits=4, token_pooling=pooling)
        idx.add(DOCS)
        raw = mv_model.encode(DOCS)
        assert idx.offsets[-1] == sum(len(m) for m in pooling.pool(raw))
        assert idx.offsets[-1] < sum(len(m) for m in raw)

    def test_same_codes_as_pooling_by_hand(self, mv_model):
        pooling = FakeTokenPooling(pool_factor=3)
        auto = RemexIndex(mv_model, bits=4, center=True, token_pooling=pooling)
        auto.add(DOCS)
        by_hand = RemexIndex(mv_model, bits=4, center=True)
        by_hand.add(pooling.pool(mv_model.encode(DOCS)))
        assert np.array_equal(auto.compressed.indices, by_hand.compressed.indices)
        assert np.array_equal(auto.offsets, by_hand.offsets)
        assert auto.search(QUERY, top_k=5) == by_hand.search(QUERY, top_k=5)

    def test_queries_are_not_pooled(self, mv_model):
        idx = RemexIndex(mv_model, bits=4, token_pooling=FakeTokenPooling())
        idx.add(DOCS)
        mv_model.calls.clear()
        idx.search(QUERY, top_k=3)
        encodes = [c for c in mv_model.calls if c[0] == "encode"]
        assert encodes and all(c[3] is None for c in encodes)

    def test_explicit_encode_argument_wins(self, mv_model):
        idx = RemexIndex(mv_model, bits=4, token_pooling=FakeTokenPooling(pool_factor=2))
        idx.add(DOCS, token_pooling=FakeTokenPooling(pool_factor=4))
        expected = FakeTokenPooling(pool_factor=4).pool(mv_model.encode(DOCS))
        assert idx.offsets[-1] == sum(len(m) for m in expected)

    def test_precomputed_matrices_are_stored_as_given(self, mv_model):
        raw = mv_model.encode(DOCS)
        idx = RemexIndex(None, bits=4, similarity="maxsim", dim=32,
                         token_pooling=FakeTokenPooling())
        idx.add(raw)
        assert idx.offsets[-1] == sum(len(m) for m in raw)

    def test_settings_dict_is_built_on_first_add(self, mv_model, fake_modules):
        idx = RemexIndex(mv_model, bits=4, token_pooling={"pool_factor": 2})
        assert isinstance(idx.token_pooling, dict)
        idx.add(DOCS)
        assert isinstance(idx.token_pooling, FakeTokenPooling)
        assert idx.token_pooling.pool_factor == 2


class TestPersistence:
    def test_round_trip_restores_pooling(self, mv_model, fake_modules, tmp_path):
        idx = RemexIndex(mv_model, bits=2, center=True,
                         token_pooling=FakeTokenPooling(pool_factor=3))
        idx.add(DOCS[:40])
        path = idx.save(tmp_path / "pooled")
        back = RemexIndex.load(path, mv_model)
        assert back.token_pooling == {"pool_factor": 3, "num_protected_tokens": 1,
                                      "class": "FakeTokenPooling"}
        assert back.search(QUERY, top_k=5) == idx.search(QUERY, top_k=5)
        back.add(DOCS[40:])
        idx.add(DOCS[40:])
        assert np.array_equal(back.offsets, idx.offsets)
        assert isinstance(back.token_pooling, FakeTokenPooling)

    def test_unpooled_index_records_nothing(self, mv_model, tmp_path):
        idx = RemexIndex(mv_model, bits=4)
        idx.add(DOCS)
        back = RemexIndex.load(idx.save(tmp_path / "plain"), mv_model)
        assert back.token_pooling is None

    def test_repr_names_the_pooling(self, mv_model):
        idx = RemexIndex(mv_model, bits=4, token_pooling=FakeTokenPooling())
        assert "pool_factor" in repr(idx)


class TestRefusals:
    def test_single_vector_index_refuses_pooling(self):
        with pytest.raises(ValueError, match="late-interaction"):
            RemexIndex(FakeModel(), bits=4, token_pooling={"pool_factor": 2})

    def test_object_without_settings_refused_at_save(self, mv_model, tmp_path):
        class Opaque:
            def pool(self, mats):
                return mats
        idx = RemexIndex(mv_model, bits=4, token_pooling=Opaque())
        idx.add(DOCS)
        with pytest.raises(TypeError, match="get_config_dict"):
            idx.save(tmp_path / "x")

    def test_unknown_module_name_refused(self, mv_model, fake_modules):
        idx = RemexIndex(mv_model, bits=4, token_pooling={"class": "NoSuchPooling"})
        with pytest.raises(ValueError, match="NoSuchPooling"):
            idx.add(DOCS)
