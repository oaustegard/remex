"""Compressed semantic search for sentence-transformers models.

``RemexIndex`` takes a model and a bit width and hides the quantizer::

    from sentence_transformers import SentenceTransformer
    from remex import RemexIndex

    index = RemexIndex(SentenceTransformer("all-MiniLM-L6-v2"), bits=4)
    index.add(docs)
    index.search("how small can an embedding get?", top_k=5)
    # [{'corpus_id': 3, 'score': 0.71}, ...]

Results use the shape of ``sentence_transformers.util.semantic_search``: a
list of ``{"corpus_id", "score"}`` dicts per query, best first. A single
string query returns one such list, as ``model.encode`` returns one vector
for a single string.

``semantic_search_remex`` is the same search behind the signature of
``sentence_transformers.util.quantization.semantic_search_faiss``, for code
that already holds embeddings.

sentence-transformers is not a dependency of remex. The model is used
through ``encode`` (``encode_query`` / ``encode_document`` when it has
them, so a model's query and document prompts apply), ``similarity_fn_name``
and ``get_embedding_dimension``; a model name string is loaded with
``SentenceTransformer`` on first use.
"""
from __future__ import annotations

import json
import time
import warnings
from typing import Any, Optional

import numpy as np

from remex.core import (
    AnisotropyWarning, CompressedVectors, Quantizer, anisotropy, corpus_mean,
)
from remex.packing import SUPPORTED_BITS

#: Similarities a remex index can rank by. Both are inner products; cosine
#: is an inner product of unit vectors, so inputs are normalized first.
SIMILARITIES = ("cosine", "dot")

#: Version of the metadata ``RemexIndex.save`` adds to the container .npz.
_FORMAT = 1

#: Upper bound on the (queries x corpus) score block held at once, in floats.
_SCORE_BLOCK_FLOATS = 1 << 24


def _load_model(name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Loading a model by name needs sentence-transformers: "
            "pip install 'remex[sentence-transformers]'. Or pass a model "
            "object, or embeddings with dim=."
        ) from exc
    return SentenceTransformer(name)


def _model_dim(model) -> Optional[int]:
    for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        fn = getattr(model, name, None)
        if fn is not None:
            dim = fn()
            if dim is not None:
                return int(dim)
    return None


def _resolve_similarity(model, similarity) -> str:
    if similarity is None and model is not None:
        similarity = getattr(model, "similarity_fn_name", None)
    if similarity is None:
        return "cosine"
    similarity = str(getattr(similarity, "value", similarity))
    if similarity not in SIMILARITIES:
        raise ValueError(
            f"remex ranks by inner product, so similarity must be one of "
            f"{SIMILARITIES}; got {similarity!r}. A model trained for "
            f"euclidean or manhattan distance needs a different index."
        )
    return similarity


def _unit_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def _is_text(inputs) -> bool:
    if isinstance(inputs, str):
        return True
    if isinstance(inputs, (list, tuple)) and inputs:
        return all(isinstance(x, str) for x in inputs)
    return False


class RemexIndex:
    """A compressed vector index over a sentence-transformers model.

    Args:
        model: A ``SentenceTransformer`` (or anything with its ``encode``),
            a model name to load, or None to add and search with
            precomputed embeddings only.
        bits: Bits per coordinate: 1, 2, 3, 4 or 8. 4 keeps recall@10
            around 0.92 of float32 on all-MiniLM-L6-v2 at 7.8x smaller;
            8 is near-lossless at 4x.
        similarity: ``"cosine"`` or ``"dot"``. Default: the model's
            ``similarity_fn_name``, else cosine.
        center: Encode relative to the mean of the first ``add`` batch.
            Pays on corpora whose vectors share one large direction (remex
            warns when it sees one); see the README's centering table.
        low_memory: Score straight from the codes instead of caching a
            float32 copy of the corpus. Same ranking, less RAM, slower
            queries.
        seed: Rotation seed. Part of the encoding; ``save`` records it.
        dim: Embedding dimension, when there is no model to ask.

    The quantizer is built on the first ``add``, so a model is never asked
    for anything until it is used.
    """

    def __init__(self, model: Any = None, bits: int = 4, *,
                 similarity: Optional[str] = None, center: bool = False,
                 low_memory: bool = False, seed: int = 42,
                 dim: Optional[int] = None):
        if bits not in SUPPORTED_BITS:
            raise ValueError(
                f"bits must be one of {list(SUPPORTED_BITS)}, got {bits}"
            )
        if isinstance(model, str):
            model = _load_model(model)
        self.model = model
        self.bits = bits
        self.similarity = _resolve_similarity(model, similarity)
        self.center = bool(center)
        self.low_memory = bool(low_memory)
        self.seed = seed
        self.dim = dim if dim is not None else (
            None if model is None else _model_dim(model)
        )
        self.quantizer: Optional[Quantizer] = None
        self.compressed: Optional[CompressedVectors] = None

    def __len__(self) -> int:
        return 0 if self.compressed is None else self.compressed.n

    def __repr__(self) -> str:
        return (
            f"RemexIndex(n={len(self)}, dim={self.dim}, bits={self.bits}, "
            f"similarity={self.similarity!r}, center={self.center})"
        )

    @property
    def nbytes(self) -> int:
        """Bit-packed size of the stored codes and norms."""
        return 0 if self.compressed is None else self.compressed.nbytes

    @property
    def compression_ratio(self) -> float:
        """float32 size of the same corpus over ``nbytes``."""
        if self.compressed is None:
            raise ValueError("the index is empty")
        return self.compressed.compression_ratio

    def _embed(self, inputs, kind: str, encode_kwargs: dict) -> np.ndarray:
        if _is_text(inputs):
            if self.model is None:
                raise ValueError(
                    "this index has no model, so it takes embeddings, not text"
                )
            texts = [inputs] if isinstance(inputs, str) else list(inputs)
            encode = getattr(self.model, f"encode_{kind}", None) or self.model.encode
            kwargs = {"convert_to_numpy": True,
                      "normalize_embeddings": self.similarity == "cosine"}
            kwargs.update(encode_kwargs)
            X = encode(texts, **kwargs)
        else:
            if encode_kwargs:
                raise TypeError(
                    f"encode arguments {sorted(encode_kwargs)} were given "
                    f"with precomputed embeddings"
                )
            X = inputs
        if hasattr(X, "detach"):  # torch tensor
            X = X.detach().cpu().numpy()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[np.newaxis]
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError(f"expected a non-empty (n, d) batch, got {X.shape}")
        if self.similarity == "cosine":
            X = _unit_rows(X)
        if self.dim is None:
            self.dim = X.shape[1]
        elif X.shape[1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {X.shape[1]}")
        return X

    def _encode(self, X: np.ndarray) -> CompressedVectors:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            codes = self.quantizer.encode(X)
        for w in caught:
            if issubclass(w.category, AnisotropyWarning):
                warnings.warn(
                    f"These {len(X)} vectors share a large common direction "
                    f"(remex.anisotropy = {anisotropy(X):.2f}), which costs "
                    f"recall at {self.bits} bits. RemexIndex(..., center=True) "
                    f"encodes relative to the corpus mean and gained up to "
                    f"0.17 recall on corpora like this.",
                    AnisotropyWarning, stacklevel=3,
                )
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        return codes

    def add(self, documents, **encode_kwargs) -> np.ndarray:
        """Encode and compress documents, returning their corpus ids.

        Args:
            documents: Strings (encoded with the model) or an (n, d) array
                of embeddings.
            **encode_kwargs: Passed to the model's encode, e.g.
                ``batch_size`` or ``show_progress_bar``.

        Ids are positions in insertion order, starting at ``len(index)``.
        Each call re-concatenates the stored codes, so add in batches
        rather than one document at a time. With ``center=True`` the mean
        is taken from the first batch, so make that the bulk of the corpus.
        """
        X = self._embed(documents, "document", encode_kwargs)
        if self.quantizer is None:
            mean = corpus_mean(X) if self.center else None
            self.quantizer = Quantizer(
                d=self.dim, bits=self.bits, seed=self.seed, mean=mean
            )
        codes = self._encode(X)
        start = len(self)
        if self.compressed is None:
            self.compressed = codes
        else:
            old = self.compressed
            self.compressed = CompressedVectors(
                np.concatenate([old.indices, codes.indices]),
                np.concatenate([old.norms, codes.norms]),
                old.d, old.bits, old.rotation, old.mean,
            )
        return np.arange(start, len(self))

    def search(self, queries, top_k: int = 10, **encode_kwargs):
        """Find the ``top_k`` best-scoring documents for each query.

        Args:
            queries: A string, strings, a (d,) embedding or (n, d) embeddings.
            top_k: Results per query.
            **encode_kwargs: Passed to the model's encode.

        Returns:
            For a single string or (d,) vector, one list of
            ``{"corpus_id": int, "score": float}`` dicts, best first.
            Otherwise one such list per query.
        """
        if self.compressed is None:
            raise ValueError("the index is empty; add documents first")
        single = isinstance(queries, str) or (
            not _is_text(queries) and np.ndim(queries) == 1
        )
        Q = self._embed(queries, "query", encode_kwargs)
        k = min(int(top_k), len(self))
        if k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        pq, codes = self.quantizer, self.compressed
        if self.low_memory:
            pairs = [pq.search_adc(codes, q, k=k) for q in Q]
            idx = np.stack([p[0] for p in pairs])
            scores = np.stack([p[1] for p in pairs])
        else:
            step = max(1, _SCORE_BLOCK_FLOATS // len(self))
            parts = [pq.search_batch(codes, Q[i:i + step], k=k)
                     for i in range(0, len(Q), step)]
            idx = np.concatenate([p[0] for p in parts])
            scores = np.concatenate([p[1] for p in parts])

        results = [
            [{"corpus_id": int(i), "score": float(s)} for i, s in zip(row_i, row_s)]
            for row_i, row_s in zip(idx, scores)
        ]
        return results[0] if single else results

    def save(self, path: str) -> str:
        """Write the index to one ``.npz`` and return its path.

        The file is a ``CompressedVectors`` container (``CompressedVectors
        .load`` reads it) plus the seed and settings needed to rebuild the
        quantizer. The model is not saved; pass it again to ``load``.
        """
        if self.compressed is None:
            raise ValueError("the index is empty; nothing to save")
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        meta = {"format": _FORMAT, "seed": self.seed,
                "similarity": self.similarity, "center": self.center}
        np.savez_compressed(
            path, remex_index=np.str_(json.dumps(meta)),
            **self.compressed._npz_fields(),
        )
        return path

    @classmethod
    def load(cls, path: str, model: Any = None, *,
             low_memory: bool = False) -> "RemexIndex":
        """Read an index written by ``save``.

        Args:
            path: The ``.npz`` file.
            model: The model the index was built with, needed to search by
                text. Its dimension is checked against the file.
            low_memory: As in the constructor; not stored in the file.
        """
        with np.load(path) as data:
            if "remex_index" not in data:
                raise ValueError(
                    f"{path} is a remex container but not a RemexIndex: it "
                    f"does not record the seed its quantizer was built with"
                )
            meta = json.loads(str(data["remex_index"]))
        if meta.get("format") != _FORMAT:
            raise ValueError(f"unsupported RemexIndex format {meta.get('format')!r}")
        codes = CompressedVectors.load(path)
        index = cls(model, bits=codes.bits, similarity=meta["similarity"],
                    center=meta["center"], low_memory=low_memory,
                    seed=meta["seed"], dim=codes.d)
        if model is not None:
            model_dim = _model_dim(index.model)
            if model_dim is not None and model_dim != codes.d:
                raise ValueError(
                    f"{path} holds dim={codes.d} vectors; the model "
                    f"produces dim={model_dim}"
                )
        index.quantizer = Quantizer(
            d=codes.d, bits=codes.bits, seed=meta["seed"],
            rotation=codes.rotation, mean=codes.mean,
        )
        index.compressed = codes
        return index


def semantic_search_remex(
    query_embeddings: np.ndarray,
    corpus_embeddings: Optional[np.ndarray] = None,
    corpus_index: Optional[RemexIndex] = None,
    bits: int = 4,
    top_k: int = 10,
    similarity: str = "cosine",
    output_index: bool = False,
):
    """``semantic_search_faiss``'s signature, with a remex index behind it.

    Pass ``corpus_embeddings`` to build an index, or a ``corpus_index``
    returned by an earlier call with ``output_index=True`` to reuse one.

    Returns:
        ``(results, search_time)``, or ``(results, search_time, index)``
        with ``output_index=True``. ``results`` holds one list of
        ``{"corpus_id", "score"}`` dicts per query; ``search_time`` is the
        seconds spent searching, excluding the build.
    """
    if corpus_index is None:
        if corpus_embeddings is None:
            raise ValueError("pass corpus_embeddings or corpus_index")
        corpus_index = RemexIndex(bits=bits, similarity=similarity)
        corpus_index.add(corpus_embeddings)
    Q = np.asarray(query_embeddings, dtype=np.float32)
    if Q.ndim == 1:
        Q = Q[np.newaxis]
    start = time.perf_counter()
    results = corpus_index.search(Q, top_k=top_k)
    search_time = time.perf_counter() - start
    if output_index:
        return results, search_time, corpus_index
    return results, search_time
