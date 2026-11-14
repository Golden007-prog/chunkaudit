"""Convert SQuAD v1.1 into a chunkaudit dataset.

SQuAD has answer positions for every question, which is exactly what an audit needs. Its
paragraphs are short, so the paragraphs of each Wikipedia article are joined into one long
document, and the answer offsets are shifted to match.

By default the evidence of a question is the whole sentence that contains the answer, because
a model needs that sentence, not just the two or three answer words, to answer reliably.

SQuAD is distributed under CC BY-SA 4.0 by Rajpurkar et al. It is downloaded on request and
is not part of this package.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from .text import sentence_spans

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
PARAGRAPH_BREAK = "\n\n"


def download_squad(cache: str | Path) -> Path:
    """Download the SQuAD v1.1 development set once and return the local path."""
    target = Path(cache) / "squad-dev-v1.1.json"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        urllib.request.urlretrieve(SQUAD_URL, partial)  # noqa: S310 - fixed https URL
        partial.replace(target)
    return target


def convert_squad(
    source: str | Path,
    out: str | Path,
    evidence: str = "sentence",
    max_docs: int | None = None,
) -> tuple[int, int]:
    """Write ``corpus.jsonl`` and ``questions.jsonl`` to ``out``.

    ``evidence`` is ``"sentence"`` (the sentence or sentences containing the answer) or
    ``"answer"`` (the answer text only). Returns the number of documents and questions written.
    """
    if evidence not in ("sentence", "answer"):
        raise ValueError("evidence must be 'sentence' or 'answer'")
    articles: list[dict[str, Any]] = json.loads(Path(source).read_text(encoding="utf-8"))["data"]
    if max_docs is not None:
        articles = articles[:max_docs]

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    documents = questions = 0
    with (
        (out_dir / "corpus.jsonl").open("w", encoding="utf-8", newline="\n") as corpus,
        (out_dir / "questions.jsonl").open("w", encoding="utf-8", newline="\n") as question_file,
    ):
        for number, article in enumerate(articles):
            doc_id = f"{number:03d}-{article['title']}"
            parts: list[str] = []
            offsets: list[int] = []
            position = 0
            for paragraph in article["paragraphs"]:
                # A blank line inside a paragraph would read as a paragraph break.
                context = paragraph["context"].replace("\n", " ")
                offsets.append(position)
                parts.append(context)
                position += len(context) + len(PARAGRAPH_BREAK)
            text = PARAGRAPH_BREAK.join(parts)
            sentences = sentence_spans(text) if evidence == "sentence" else []
            corpus.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")
            documents += 1

            for offset, paragraph in zip(offsets, article["paragraphs"], strict=True):
                for qa in paragraph["qas"]:
                    if not qa["answers"]:
                        continue
                    answer = qa["answers"][0]
                    start = offset + int(answer["answer_start"])
                    end = start + len(answer["text"])
                    if text[start:end] != answer["text"]:
                        continue  # the published offset does not match the text
                    if evidence == "sentence":
                        start, end = _enclosing(sentences, start, end)
                    row = {
                        "id": qa["id"],
                        "question": qa["question"].strip(),
                        "doc_id": doc_id,
                        "evidence": text[start:end],
                        "evidence_start": start,
                    }
                    question_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    questions += 1
    return documents, questions


def _enclosing(sentences: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    """Smallest run of sentences that covers ``[start, end)``."""
    overlapping = [span for span in sentences if span[0] < end and span[1] > start]
    if not overlapping:
        return start, end
    return min(start, overlapping[0][0]), max(end, overlapping[-1][1])
