"""The embedding retriever is tested with a small deterministic encoder, so no model is needed."""

import numpy as np
import pytest

from chunkaudit import (
    EmbeddingIndex,
    EmbeddingRetriever,
    SentenceChunker,
    compare,
    evaluate,
    load_dataset,
    parse_strategy,
    sample_path,
)
from chunkaudit import embeddings as embeddings_module
from chunkaudit.cli import main
from chunkaudit.text import tokenize

DIMENSIONS = 64


def bag_of_words(texts):
    """One row per text: counts of its tokens, hashed into a fixed number of columns."""
    matrix = np.zeros((len(texts), DIMENSIONS), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in tokenize(text):
            matrix[row, sum(map(ord, token)) % DIMENSIONS] += 1.0
    return matrix


TEXTS = [
    "the moon pulls the oceans and causes tides",
    "bread rises because yeast makes carbon dioxide",
    "bees dance to share where the flowers are",
]


def test_index_ranks_by_cosine_similarity():
    index = EmbeddingIndex(TEXTS, bag_of_words)
    assert index.size == 3
    assert index.search("moon tides oceans", 1) == [0]
    assert index.search("yeast bread", 2)[0] == 1
    assert sorted(index.search("anything", 10)) == [0, 1, 2]


def test_index_ties_break_by_position_and_edge_cases():
    index = EmbeddingIndex(["same words", "same words", "same words"], bag_of_words)
    assert index.search("same words", 2) == [0, 1]
    assert index.search("same words", 0) == []
    assert EmbeddingIndex([], bag_of_words).search("anything", 3) == []
    # A text with no tokens has a zero vector; it must not produce NaN scores.
    assert EmbeddingIndex(["", "moon"], bag_of_words).search("moon", 1) == [1]


def test_retriever_embeds_each_question_once():
    calls = []

    def counting(texts):
        calls.append(list(texts))
        return bag_of_words(texts)

    retriever = EmbeddingRetriever(encode=counting)
    retriever.warm(["moon tides", "yeast bread", "moon tides"])
    assert calls == [["moon tides", "yeast bread"]]

    first, second = retriever(TEXTS), retriever(TEXTS[:2])
    assert first.search("moon tides", 1) == [0]
    assert second.search("yeast bread", 1) == [1]
    assert first.search("bees flowers", 1) == [2]
    # two corpus encodings, then one new query; the warmed queries were not embedded again
    assert calls[1:] == [TEXTS, TEXTS[:2], ["bees flowers"]]


def test_evaluate_and_compare_accept_the_retriever():
    dataset = load_dataset(sample_path())
    retriever = EmbeddingRetriever(encode=bag_of_words)
    retriever.warm([question.text for question in dataset.questions])
    result = evaluate(dataset, SentenceChunker(2), retriever=retriever)
    assert result.split_rate == 0.0
    assert 0.0 < result.recall[5] <= 1.0

    fixed, sentence = compare(
        dataset, [parse_strategy("fixed:40"), parse_strategy("sentence:2")], retriever=retriever
    )
    # The split rate depends on the chunker only, so it matches the BM25 audit.
    assert fixed.split_rate == 0.6
    assert sentence.recall[5] >= fixed.recall[5]


def test_missing_library_gives_an_install_hint(monkeypatch, capsys):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(embeddings_module.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match=r"chunkaudit\[embeddings\]"):
        EmbeddingRetriever()

    assert main(["demo", "--retriever", "embeddings"]) == 2
    assert "sentence-transformers" in capsys.readouterr().err


def test_cli_uses_the_embedding_retriever(monkeypatch, capsys):
    class FakeModel:
        def __init__(self, name):
            self.name = name

        def encode(self, texts, batch_size, show_progress_bar):
            return bag_of_words(texts)

    class FakeLibrary:
        SentenceTransformer = FakeModel

    monkeypatch.setattr(embeddings_module.importlib, "import_module", lambda name: FakeLibrary)
    code = main(["demo", "-s", "fixed:40,sentence:2", "--retriever", "embeddings"])
    captured = capsys.readouterr()
    assert code == 0
    assert "retriever: embeddings (sentence-transformers/all-MiniLM-L6-v2)" in captured.err
    assert "sentence:2" in captured.out and "fixed:40" in captured.out
