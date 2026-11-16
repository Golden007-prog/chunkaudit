"""Embedding-based retrieval, as an alternative to BM25.

BM25 ranks chunks by shared words. An embedding model ranks them by meaning, so it can rank
the same chunking strategies differently. This module provides a dense retriever with the same
``search(query, k)`` interface as :class:`chunkaudit.BM25Index`.

The model library is optional::

    pip install "chunkaudit[embeddings]"

``EmbeddingIndex`` itself only needs an ``encode`` function that turns texts into vectors, so
any embedding model, local or hosted, can be plugged in.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

Encoder = Callable[[Sequence[str]], np.ndarray]


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale every row to unit length, so a dot product is a cosine similarity."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class EmbeddingIndex:
    """Rank texts by cosine similarity between their embedding and the query's."""

    def __init__(self, texts: Sequence[str], encode: Encoder) -> None:
        self.size = len(texts)
        self._encode = encode
        self._matrix = (
            _normalise(encode(list(texts))) if texts else np.zeros((0, 0), dtype=np.float32)
        )

    def search(self, query: str, k: int) -> list[int]:
        """Indices of the ``k`` most similar texts, best first. Ties break by position."""
        if not self.size or k < 1:
            return []
        scores = self._matrix @ _normalise(self._encode([query]))[0]
        k = min(k, self.size)
        if k < self.size:
            cutoff = np.partition(scores, self.size - k)[self.size - k]
            candidates = np.flatnonzero(scores >= cutoff)
        else:
            candidates = np.arange(self.size)
        order = np.lexsort((candidates, -scores[candidates]))[:k]
        return [int(index) for index in candidates[order]]


class EmbeddingRetriever:
    """Builds :class:`EmbeddingIndex` objects that share one model and one query cache.

    Pass an instance as ``retriever=`` to :func:`chunkaudit.evaluate` or
    :func:`chunkaudit.compare`. When several strategies are audited on the same questions, each
    question is embedded once.

    ``encode`` is any function from a list of texts to a 2-D array with one row per text. Leave
    it out to load a sentence-transformers model by name.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        encode: Encoder | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self._encode_texts: Encoder = encode if encode is not None else self._load(model)
        self._queries: dict[str, np.ndarray] = {}

    def _load(self, model: str) -> Encoder:
        try:
            library: Any = importlib.import_module("sentence_transformers")
        except ImportError:
            raise RuntimeError(
                "the embedding retriever needs the sentence-transformers package: "
                'pip install "chunkaudit[embeddings]"'
            ) from None
        loaded = library.SentenceTransformer(model)

        def encode(texts: Sequence[str]) -> np.ndarray:
            vectors = loaded.encode(
                list(texts), batch_size=self.batch_size, show_progress_bar=False
            )
            return np.asarray(vectors, dtype=np.float32)

        return encode

    def warm(self, queries: Sequence[str]) -> None:
        """Embed many queries in one batch so later searches are lookups."""
        missing = [query for query in dict.fromkeys(queries) if query not in self._queries]
        if missing:
            vectors = _normalise(self._encode_texts(missing))
            self._queries.update(zip(missing, vectors, strict=True))

    def _encode_cached(self, texts: Sequence[str]) -> np.ndarray:
        """Encoder handed to each index: single queries come from the cache."""
        if len(texts) == 1:
            query = texts[0]
            if query not in self._queries:
                self.warm([query])
            return self._queries[query].reshape(1, -1)
        return self._encode_texts(texts)

    def __call__(self, texts: Sequence[str]) -> EmbeddingIndex:
        return EmbeddingIndex(texts, self._encode_cached)
