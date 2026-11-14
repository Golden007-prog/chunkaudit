"""A small BM25 index over chunk texts, built on an inverted index with NumPy.

BM25 is used as the retriever because it needs no model download and no API key, and it is a
strong lexical baseline. Anything with a ``search(query, k)`` method can replace it.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np

from .text import tokenize


class BM25Index:
    """Okapi BM25 with the per-posting weights computed once at build time."""

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.size = len(texts)
        counts = [Counter(tokenize(text)) for text in texts]
        lengths = np.array([sum(count.values()) for count in counts], dtype=np.float64)
        average = float(lengths.mean()) if self.size and lengths.sum() > 0 else 1.0
        norm = k1 * (1.0 - b + b * lengths / average)

        rows: dict[str, list[int]] = defaultdict(list)
        frequencies: dict[str, list[int]] = defaultdict(list)
        for row, count in enumerate(counts):
            for term, frequency in count.items():
                rows[term].append(row)
                frequencies[term].append(frequency)

        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, term_rows in rows.items():
            index = np.asarray(term_rows, dtype=np.int64)
            tf = np.asarray(frequencies[term], dtype=np.float64)
            idf = math.log(1.0 + (self.size - len(term_rows) + 0.5) / (len(term_rows) + 0.5))
            self._postings[term] = (index, idf * tf * (k1 + 1.0) / (tf + norm[index]))

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.size, dtype=np.float64)
        for term in set(tokenize(query)):
            posting = self._postings.get(term)
            if posting is not None:
                scores[posting[0]] += posting[1]
        return scores

    def search(self, query: str, k: int) -> list[int]:
        """Indices of the best ``k`` texts, best first. Texts with no matching term are left out.

        Ties are broken by position, so results are deterministic.
        """
        scores = self.scores(query)
        matched = np.flatnonzero(scores > 0)
        if matched.size > k:
            cutoff = np.partition(scores[matched], matched.size - k)[matched.size - k]
            matched = matched[scores[matched] >= cutoff]
        order = np.lexsort((matched, -scores[matched]))[:k]
        return [int(index) for index in matched[order]]
