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

    def encode(self, sentences, convert_to_numpy=True, **kwargs):
        self.calls.append(("encode", len(sentences), kwargs))
        return [np.stack([self._tok(w) for w in s.split()]) for s in sentences]

    def encode_query(self, sentences, **kwargs):
        self.calls.append(("query", len(sentences)))
        return self.encode(sentences, **kwargs)

    def encode_document(self, sentences, **kwargs):
        self.calls.append(("document", len(sentences)))
        return self.encode(sentences, **kwargs)
