"""Audit a chunking strategy against questions whose evidence location is known.

For every question the audit asks two things:

1. Did chunking keep the evidence in one piece? If no single chunk contains the whole evidence
   span, the answer was *split*, and no retriever can return it intact.
2. Does retrieval find a chunk that contains the whole evidence span, and how high does it rank?
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .bm25 import BM25Index
from .chunkers import Chunker
from .dataset import Dataset
from .text import count_words


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[int]: ...


RetrieverFactory = Callable[[Sequence[str]], Retriever]


@dataclass(frozen=True)
class StrategyResult:
    """Scores of one chunking strategy on one dataset."""

    strategy: str
    chunks: int
    avg_words: float
    split_rate: float
    recall: dict[int, float]
    mrr: float
    context_words: dict[int, float]
    seconds: float


def evaluate(
    dataset: Dataset,
    chunker: Chunker,
    ks: Iterable[int] = (1, 3, 5),
    mrr_depth: int = 10,
    retriever: RetrieverFactory = BM25Index,
) -> StrategyResult:
    """Chunk every document with ``chunker``, index the chunks and score every question.

    ``recall[k]`` is the share of questions with a top-``k`` chunk that contains the whole
    evidence span. ``mrr`` is the mean reciprocal rank of the first such chunk within the top
    ``mrr_depth``. ``split_rate`` is the share of questions whose evidence is not fully inside
    any chunk. ``context_words[k]`` is the mean number of words in the top ``k`` chunks, which
    is what would be sent to a language model.
    """
    cutoffs = sorted(set(ks))
    if not cutoffs or cutoffs[0] < 1 or mrr_depth < 1:
        raise ValueError("ks and mrr_depth must be positive")
    began = time.perf_counter()

    texts: list[str] = []
    words: list[int] = []
    by_document: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for document in dataset.documents:
        spans = chunker.split(document.text)
        ids = np.arange(len(texts), len(texts) + len(spans))
        starts = np.array([start for start, _ in spans], dtype=np.int64)
        ends = np.array([end for _, end in spans], dtype=np.int64)
        by_document[document.id] = (ids, starts, ends)
        for start, end in spans:
            texts.append(document.text[start:end])
            words.append(count_words(document.text, start, end))

    index = retriever(texts)
    word_counts = np.asarray(words, dtype=np.float64)
    depth = max(cutoffs[-1], mrr_depth)

    split = 0
    hits = dict.fromkeys(cutoffs, 0)
    context = dict.fromkeys(cutoffs, 0.0)
    reciprocal_ranks = 0.0
    for question in dataset.questions:
        ids, starts, ends = by_document[question.doc_id]
        containing = set(ids[(starts <= question.start) & (ends >= question.end)].tolist())
        if not containing:
            split += 1
        ranked = index.search(question.text, depth)
        rank = next((position for position, chunk in enumerate(ranked) if chunk in containing), -1)
        if 0 <= rank < mrr_depth:
            reciprocal_ranks += 1.0 / (rank + 1)
        for k in cutoffs:
            if 0 <= rank < k:
                hits[k] += 1
            context[k] += float(word_counts[ranked[:k]].sum())

    total = len(dataset.questions)
    return StrategyResult(
        strategy=chunker.name,
        chunks=len(texts),
        avg_words=float(word_counts.mean()) if texts else 0.0,
        split_rate=split / total,
        recall={k: hits[k] / total for k in cutoffs},
        mrr=reciprocal_ranks / total,
        context_words={k: context[k] / total for k in cutoffs},
        seconds=time.perf_counter() - began,
    )


@dataclass(frozen=True)
class SplitExample:
    """One question whose evidence a chunking strategy failed to keep in a single chunk."""

    question_id: str
    question: str
    doc_id: str
    evidence: str
    """The evidence text, with ``marker`` inserted wherever a chunk ends inside it."""
    pieces: int
    """Number of chunks the evidence is spread across."""


def split_examples(
    dataset: Dataset,
    chunker: Chunker,
    limit: int | None = None,
    marker: str = " || ",
) -> list[SplitExample]:
    """Questions whose evidence is not fully inside any chunk, in dataset order.

    Use this to see what a strategy cuts, not only how often: each example shows the evidence
    with ``marker`` at the positions where a chunk ends.
    """
    spans = {document.id: chunker.split(document.text) for document in dataset.documents}
    texts = {document.id: document.text for document in dataset.documents}
    examples: list[SplitExample] = []
    for question in dataset.questions:
        if limit is not None and len(examples) >= limit:
            break
        chunks = spans[question.doc_id]
        if any(start <= question.start and end >= question.end for start, end in chunks):
            continue
        text = texts[question.doc_id]
        cuts = sorted({end for _, end in chunks if question.start < end < question.end})
        edges = [question.start, *cuts, question.end]
        parts = [text[left:right].strip() for left, right in zip(edges, edges[1:], strict=False)]
        examples.append(
            SplitExample(
                question_id=question.id,
                question=question.text,
                doc_id=question.doc_id,
                evidence=marker.join(part for part in parts if part),
                pieces=sum(1 for s, e in chunks if s < question.end and e > question.start),
            )
        )
    return examples


def compare(
    dataset: Dataset,
    chunkers: Iterable[Chunker],
    ks: Iterable[int] = (1, 3, 5),
    mrr_depth: int = 10,
    retriever: RetrieverFactory = BM25Index,
) -> list[StrategyResult]:
    """Audit several strategies on the same dataset, in the order given."""
    cutoffs = tuple(ks)
    return [evaluate(dataset, chunker, cutoffs, mrr_depth, retriever) for chunker in chunkers]
