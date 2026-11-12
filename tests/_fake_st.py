"""A stand-in for ``sentence_transformers.SentenceTransformer``.

CI installs only ``.[dev]``, never torch, so the integration is tested
against this: deterministic vectors per string, and the parts of the real
model's surface that ``remex.st`` touches.
"""
import zlib

import numpy as np


class FakeModel:
    def __init__(self, name="fake", dim=64, similarity_fn_name="cosine"):
        self.name = name
        self.dim = dim
        self.similarity_fn_name = similarity_fn_name
        self.calls = []

    def get_embedding_dimension(self):
        return self.dim

    def _vec(self, text):
        rng = np.random.default_rng(zlib.crc32(text.encode("utf-8")))
        return rng.standard_normal(self.dim).astype(np.float32)

    def encode(self, sentences, convert_to_numpy=True,
               normalize_embeddings=False, **kwargs):
        self.calls.append(("encode", len(sentences), normalize_embeddings, kwargs))
        X = np.stack([self._vec(s) for s in sentences])
        if normalize_embeddings:
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def encode_query(self, sentences, **kwargs):
        self.calls.append(("query", len(sentences)))
        return self.encode(sentences, **kwargs)

    def encode_document(self, sentences, **kwargs):
        self.calls.append(("document", len(sentences)))
        return self.encode(sentences, **kwargs)


class SentenceTransformer(FakeModel):
    """Constructed by name, like the real one."""

    def __init__(self, model_name_or_path="fake", **kwargs):
        super().__init__(name=model_name_or_path)


class FakeMultiVectorModel:
    """Stands in for ``sentence_transformers.MultiVectorEncoder``: one
    deterministic unit vector per word, so a document is a (words, dim)
    matrix and repeated words repeat their vector."""

    similarity_fn_name = "maxsim"

    def __init__(self, dim=32, shared=0.0):
        self.dim = dim
        self.shared = shared  # weight of one direction every token shares
        self.calls = []

    def get_embedding_dimension(self):
        return self.dim

    def _tok(self, word):
        rng = np.random.default_rng(zlib.crc32(word.encode("utf-8")))
        v = rng.standard_normal(self.dim) + self.shared
        return (v / np.linalg.norm(v)).astype(np.float32)

    def encode(self, sentences, convert_to_numpy=True, token_pooling=None, **kwargs):
        self.calls.append(("encode", len(sentences), kwargs, token_pooling))
        mats = [np.stack([self._tok(w) for w in s.split()]) for s in sentences]
        return mats if token_pooling is None else token_pooling.pool(mats)

    def encode_query(self, sentences, **kwargs):
        self.calls.append(("query", len(sentences)))
        return self.encode(sentences, **kwargs)

    def encode_document(self, sentences, **kwargs):
        self.calls.append(("document", len(sentences)))
        return self.encode(sentences, **kwargs)


class FakeTokenPooling:
    """Stands in for ``HierarchicalTokenPooling``: keeps the first token and
    replaces each later run of ``pool_factor`` tokens with its renormalised
    mean, and reports its settings the way sentence-transformers modules do."""

    config_keys = ["pool_factor", "num_protected_tokens"]

    def __init__(self, pool_factor=2, num_protected_tokens=1, **kwargs):
        self.pool_factor = pool_factor
        self.num_protected_tokens = num_protected_tokens

    def get_config_dict(self):
        return {k: getattr(self, k) for k in self.config_keys}

    def pool(self, mats):
        out = []
        for m in mats:
            keep, rest = m[:self.num_protected_tokens], m[self.num_protected_tokens:]
            groups = [rest[i:i + self.pool_factor] for i in range(0, len(rest), self.pool_factor)]
            means = [g.mean(0) for g in groups]
            means = [v / np.linalg.norm(v) for v in means]
            out.append(np.vstack([keep] + ([np.stack(means)] if means else [])).astype(np.float32))
        return out
