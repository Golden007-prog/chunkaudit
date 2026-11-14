"""Text segmentation helpers.

Every function returns character spans ``(start, end)`` into the original text, so a chunk can
always be traced back to the exact characters it covers.
"""

from __future__ import annotations

import re

Span = tuple[int, int]

_WORD = re.compile(r"\S+")
_TOKEN = re.compile(r"[a-z0-9]+")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")
# Sentence-ending punctuation, optional closing quotes or brackets, whitespace, and then
# something that looks like the start of a new sentence.
_SENTENCE_END = re.compile(r"""[.!?]+["')\]]*\s+(?=["'(\[]?[A-Z0-9])""")
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "e.g", "i.e",
        "fig", "inc", "ltd", "gen", "col", "lt", "sgt", "capt", "rev", "u.s", "u.k",
    }
)  # fmt: skip


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens, used for retrieval."""
    return _TOKEN.findall(text.lower())


def word_spans(text: str, start: int = 0, end: int | None = None) -> list[Span]:
    """Spans of whitespace-separated words inside ``text[start:end]``."""
    stop = len(text) if end is None else end
    return [match.span() for match in _WORD.finditer(text, start, stop)]


def count_words(text: str, start: int = 0, end: int | None = None) -> int:
    stop = len(text) if end is None else end
    return sum(1 for _ in _WORD.finditer(text, start, stop))


def paragraph_spans(text: str) -> list[Span]:
    """Spans of paragraphs, which are separated by one or more blank lines."""
    spans: list[Span] = []
    position = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        spans.append((position, match.start()))
        position = match.end()
    spans.append((position, len(text)))
    return [span for span in (_trim(text, start, end) for start, end in spans) if span]


def sentence_spans(text: str) -> list[Span]:
    """Spans of sentences. A sentence never crosses a paragraph break.

    The splitter is rule based: it breaks after ``.``, ``!`` or ``?`` when the next word starts
    with a capital letter or a digit, and it skips common abbreviations and single initials.
    """
    spans: list[Span] = []
    for paragraph_start, paragraph_end in paragraph_spans(text):
        spans.extend(sentences_in(text, paragraph_start, paragraph_end))
    return spans


def sentences_in(text: str, start: int, end: int) -> list[Span]:
    """Spans of the sentences inside one paragraph, ``text[start:end]``."""
    spans: list[Span] = []
    for match in _SENTENCE_END.finditer(text, start, end):
        if _ends_with_abbreviation(text[start : match.start()]):
            continue
        spans.append((start, match.start() + len(match.group().rstrip())))
        start = match.end()
    if start < end:
        spans.append((start, end))
    return spans


def _ends_with_abbreviation(fragment: str) -> bool:
    words = fragment.split()
    if not words:
        return False
    last = words[-1].lstrip("\"'([").lower()
    return last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha())


def _trim(text: str, start: int, end: int) -> Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None
