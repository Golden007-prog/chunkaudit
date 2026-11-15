import pytest

from chunkaudit import (
    Dataset,
    Document,
    FixedChunker,
    Question,
    SentenceChunker,
    compare,
    evaluate,
    load_dataset,
    parse_strategy,
    sample_path,
    split_examples,
)

TEXT = (
    "Tides follow the moon closely. Bread needs yeast to rise well. "
    "Bees dance to share flowers. Lighthouses guide ships at night."
)


def make_dataset():
    def span(evidence):
        start = TEXT.index(evidence)
        return start, start + len(evidence)

    questions = [
        Question("q1", "what follows the moon", "doc", *span("Tides follow the moon closely.")),
        Question("q2", "what does bread need", "doc", *span("Bread needs yeast to rise well.")),
        Question("q3", "why do bees dance", "doc", *span("Bees dance to share flowers.")),
        Question("q4", "what guides ships", "doc", *span("Lighthouses guide ships at night.")),
    ]
    return Dataset("toy", [Document("doc", TEXT)], questions)


def test_sentence_chunks_keep_every_answer_whole():
    result = evaluate(make_dataset(), SentenceChunker(1), ks=(1, 2))
    assert result.strategy == "sentence:1"
    assert result.chunks == 4
    assert result.split_rate == 0.0
    assert result.recall == {1: 1.0, 2: 1.0}
    assert result.mrr == 1.0
    assert result.avg_words == pytest.approx(21 / 4)


def test_fixed_chunks_that_cut_sentences_split_answers():
    # 4 words per chunk: only "Bees dance to share" ... never holds a full five-word sentence.
    result = evaluate(make_dataset(), FixedChunker(4), ks=(1, 3))
    assert result.split_rate == 1.0
    assert result.recall == {1: 0.0, 3: 0.0}
    assert result.mrr == 0.0


def test_partial_split_is_counted_per_question():
    # The sentences are 5, 6, 5 and 5 words long. Chunks of 7 words cover words 1-7, 8-14 and
    # 15-21, so the first and last sentences stay whole and the two in the middle are cut.
    result = evaluate(make_dataset(), FixedChunker(7), ks=(1,))
    assert result.chunks == 3
    assert result.split_rate == 0.5
    assert result.recall[1] == 0.5
    assert result.mrr == 0.5


def test_one_big_chunk_always_contains_the_answer_but_costs_more_words():
    whole = evaluate(make_dataset(), FixedChunker(1000), ks=(1,))
    small = evaluate(make_dataset(), SentenceChunker(1), ks=(1,))
    assert whole.chunks == 1 and whole.split_rate == 0.0 and whole.recall[1] == 1.0
    assert whole.context_words[1] == 21
    assert small.context_words[1] < whole.context_words[1]


def test_a_wrong_document_chunk_does_not_count():
    dataset = make_dataset()
    other = Document("other", "Tides follow the moon closely in this other text about the moon.")
    mixed = Dataset("toy", [other, *dataset.documents], dataset.questions)
    result = evaluate(mixed, FixedChunker(1000), ks=(1, 2))
    # question 1 ranks the look-alike document first, so it only hits at k=2
    assert result.recall[1] == 0.75
    assert result.recall[2] == 1.0
    assert result.mrr == pytest.approx((0.5 + 3) / 4)


def test_a_custom_retriever_can_be_plugged_in():
    class LastChunkFirst:
        def __init__(self, texts):
            self.size = len(texts)

        def search(self, query, k):
            return list(range(self.size - 1, -1, -1))[:k]

    result = evaluate(make_dataset(), SentenceChunker(1), ks=(1, 4), retriever=LastChunkFirst)
    assert result.recall == {1: 0.25, 4: 1.0}
    assert result.mrr == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 4) / 4)


def test_compare_keeps_the_order_and_validates_cutoffs():
    results = compare(make_dataset(), [FixedChunker(4), SentenceChunker(1)], ks=(3, 1, 3))
    assert [result.strategy for result in results] == ["fixed:4", "sentence:1"]
    assert list(results[0].recall) == [1, 3]
    with pytest.raises(ValueError):
        evaluate(make_dataset(), SentenceChunker(1), ks=())
    with pytest.raises(ValueError):
        evaluate(make_dataset(), SentenceChunker(1), ks=(0,))


def test_sample_dataset_shows_the_expected_pattern():
    dataset = load_dataset(sample_path())
    fixed, sentence = compare(dataset, [parse_strategy("fixed:40"), parse_strategy("sentence:2")])
    assert sentence.split_rate == 0.0
    assert fixed.split_rate > 0.3
    assert sentence.recall[5] > fixed.recall[5]
    for result in (fixed, sentence):
        assert 0.0 <= result.mrr <= 1.0
        assert result.recall[1] <= result.recall[3] <= result.recall[5] <= 1 - result.split_rate


def test_split_examples_mark_where_the_chunk_ends():
    # Chunks of 7 words cut the second and third sentences (see the test above).
    examples = split_examples(make_dataset(), FixedChunker(7))
    assert [example.question_id for example in examples] == ["q2", "q3"]
    assert examples[0].evidence == "Bread needs || yeast to rise well."
    assert examples[1].evidence == "Bees dance to || share flowers."
    assert all(example.pieces == 2 and example.doc_id == "doc" for example in examples)
    assert examples[0].question == "what does bread need"


def test_split_examples_respect_limit_marker_and_clean_strategies():
    dataset = make_dataset()
    assert split_examples(dataset, SentenceChunker(1)) == []
    assert len(split_examples(dataset, FixedChunker(4), limit=1)) == 1
    marked = split_examples(dataset, FixedChunker(7), marker=" <CUT> ")[0].evidence
    assert marked == "Bread needs <CUT> yeast to rise well."


def test_split_examples_with_overlap():
    # 7-word chunks stepping 4 words cover words 1-7, 5-11, 9-15, 13-19 and 17-21.
    # The second sentence (words 6-11) now fits in one chunk. The third (words 12-16) does
    # not: the chunk covering words 9-15 ends one word before the sentence does.
    examples = split_examples(make_dataset(), FixedChunker(7, overlap=3))
    assert [example.question_id for example in examples] == ["q3"]
    assert examples[0].evidence == "Bees dance to share || flowers."
    assert examples[0].pieces == 2
