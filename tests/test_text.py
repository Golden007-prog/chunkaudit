from chunkaudit.text import (
    count_words,
    paragraph_spans,
    sentence_spans,
    tokenize,
    word_spans,
)


def pieces(text, spans):
    return [text[start:end] for start, end in spans]


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Hello, World! It's 2026.") == ["hello", "world", "it", "s", "2026"]


def test_word_spans_point_at_the_words():
    text = "  one two\nthree  "
    assert pieces(text, word_spans(text)) == ["one", "two", "three"]
    assert count_words(text) == 3
    assert count_words(text, 0, 6) == 1


def test_paragraphs_are_separated_by_blank_lines():
    text = "First paragraph.\nStill first.\n\nSecond.\n \n\n  Third.  \n"
    assert pieces(text, paragraph_spans(text)) == [
        "First paragraph.\nStill first.",
        "Second.",
        "Third.",
    ]


def test_sentences_split_on_end_punctuation():
    text = 'It rained. "Did it stop?" she asked. Yes! It did.'
    assert pieces(text, sentence_spans(text)) == [
        "It rained.",
        '"Did it stop?" she asked.',
        "Yes!",
        "It did.",
    ]


def test_sentences_keep_abbreviations_and_initials_together():
    text = "Dr. Smith met J. K. Rowling in the U.S. Army museum. They talked, e.g. about books."
    assert pieces(text, sentence_spans(text)) == [
        "Dr. Smith met J. K. Rowling in the U.S. Army museum.",
        "They talked, e.g. about books.",
    ]


def test_sentences_do_not_split_on_decimals_or_lowercase():
    text = "The value is 3.14 exactly. it stays one sentence. Really."
    assert pieces(text, sentence_spans(text)) == [
        "The value is 3.14 exactly. it stays one sentence.",
        "Really.",
    ]


def test_sentences_never_cross_a_paragraph_break():
    text = "A heading without a full stop\n\nThe body starts here. It has two sentences."
    assert pieces(text, sentence_spans(text)) == [
        "A heading without a full stop",
        "The body starts here.",
        "It has two sentences.",
    ]


def test_empty_text_has_no_spans():
    assert word_spans("") == []
    assert paragraph_spans("  \n\n ") == []
    assert sentence_spans("") == []
