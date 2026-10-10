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
