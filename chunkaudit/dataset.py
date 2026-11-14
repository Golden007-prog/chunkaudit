"""Dataset format and loader.

A dataset is a folder with two JSON Lines files:

``corpus.jsonl``, one document per line::

    {"id": "tides", "text": "Tides are the regular rise and fall of sea level ..."}

``questions.jsonl``, one question per line::

    {"id": "q1", "question": "What mainly causes tides?", "doc_id": "tides",
     "evidence": "They are caused mainly by the gravitational pull of the Moon"}

``evidence`` is the exact text in the document that answers the question. It is located by
exact match. When the same text occurs more than once, add ``"evidence_start": <character
offset>`` to say which occurrence is meant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DatasetError(ValueError):
    """The dataset files are missing, malformed or inconsistent."""


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class Question:
    """A question and the character span of its evidence inside one document."""

    id: str
    text: str
    doc_id: str
    start: int
    end: int


@dataclass(frozen=True)
class Dataset:
    name: str
    documents: list[Document]
    questions: list[Question]


def sample_path() -> Path:
    """Folder of the small dataset that ships with the package."""
    return Path(__file__).resolve().parent / "data" / "sample"


def load_dataset(path: str | Path) -> Dataset:
    folder = Path(path)
    documents = [
        Document(id=str(_field(row, "id", where)), text=str(_field(row, "text", where)))
        for row, where in _read_jsonl(folder / "corpus.jsonl")
    ]
    texts = {document.id: document.text for document in documents}
    if len(texts) != len(documents):
        raise DatasetError(f"{folder / 'corpus.jsonl'}: document ids must be unique")

    questions = []
    for row, where in _read_jsonl(folder / "questions.jsonl"):
        doc_id = str(_field(row, "doc_id", where))
        if doc_id not in texts:
            raise DatasetError(f"{where}: doc_id {doc_id!r} is not in corpus.jsonl")
        start, end = _locate(texts[doc_id], row, where)
        questions.append(
            Question(
                id=str(_field(row, "id", where)),
                text=str(_field(row, "question", where)),
                doc_id=doc_id,
                start=start,
                end=end,
            )
        )
    if not documents or not questions:
        raise DatasetError(f"{folder}: the dataset needs at least one document and one question")
    return Dataset(name=folder.name, documents=documents, questions=questions)


def _locate(text: str, row: dict[str, Any], where: str) -> tuple[int, int]:
    evidence = str(_field(row, "evidence", where))
    if not evidence:
        raise DatasetError(f"{where}: evidence is empty")
    if "evidence_start" in row:
        start = int(row["evidence_start"])
        if text[start : start + len(evidence)] != evidence:
            raise DatasetError(f"{where}: evidence is not at evidence_start {start}")
    else:
        start = text.find(evidence)
        if start == -1:
            raise DatasetError(f"{where}: evidence was not found in document {row['doc_id']!r}")
    return start, start + len(evidence)


def _field(row: dict[str, Any], name: str, where: str) -> Any:
    if name not in row:
        raise DatasetError(f"{where}: missing field {name!r}")
    return row[name]


def _read_jsonl(path: Path) -> list[tuple[dict[str, Any], str]]:
    if not path.is_file():
        raise DatasetError(f"{path}: file not found")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            where = f"{path}:{number}"
            try:
                row = json.loads(line)
            except ValueError as error:
                raise DatasetError(f"{where}: invalid JSON ({error})") from None
            if not isinstance(row, dict):
                raise DatasetError(f"{where}: each line must be a JSON object")
            rows.append((row, where))
    return rows
