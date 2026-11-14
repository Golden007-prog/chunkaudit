import json

import pytest

from chunkaudit import DatasetError, load_dataset, sample_path


def write_dataset(folder, documents, questions):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "corpus.jsonl").write_text(
        "\n".join(json.dumps(row) for row in documents) + "\n", encoding="utf-8"
    )
    (folder / "questions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in questions) + "\n", encoding="utf-8"
    )
    return folder


DOCUMENT = {"id": "d1", "text": "The cat sat. The dog ran. The cat sat again."}


def question(**overrides):
    row = {"id": "q1", "question": "Who ran?", "doc_id": "d1", "evidence": "The dog ran."}
    row.update(overrides)
    return row


def test_sample_dataset_loads_and_evidence_spans_are_exact():
    dataset = load_dataset(sample_path())
    assert dataset.name == "sample"
    assert len(dataset.documents) == 5
    assert len(dataset.questions) == 20
    texts = {document.id: document.text for document in dataset.documents}
    for item in dataset.questions:
        evidence = texts[item.doc_id][item.start : item.end]
        assert evidence and evidence == evidence.strip()
        assert evidence[0].isupper() and evidence.endswith(".")


def test_evidence_is_located_by_exact_match(tmp_path):
    dataset = load_dataset(write_dataset(tmp_path / "data", [DOCUMENT], [question()]))
    item = dataset.questions[0]
    assert (item.start, item.end) == (13, 25)
    assert DOCUMENT["text"][item.start : item.end] == "The dog ran."


def test_evidence_start_selects_a_later_occurrence(tmp_path):
    rows = [question(evidence="The cat sat", evidence_start=26)]
    dataset = load_dataset(write_dataset(tmp_path / "data", [DOCUMENT], rows))
    assert dataset.questions[0].start == 26


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (question(evidence="The bird flew."), "was not found"),
        (question(evidence="The cat sat", evidence_start=3), "not at evidence_start"),
        (question(doc_id="missing"), "not in corpus.jsonl"),
        (question(evidence=""), "evidence is empty"),
        ({"id": "q1", "question": "Who ran?", "doc_id": "d1"}, "missing field 'evidence'"),
    ],
)
def test_bad_questions_are_reported_with_file_and_line(tmp_path, row, message):
    folder = write_dataset(tmp_path / "data", [DOCUMENT], [row])
    with pytest.raises(DatasetError, match=message) as error:
        load_dataset(folder)
    assert "questions.jsonl:1" in str(error.value)


def test_missing_files_duplicate_ids_and_bad_json_are_reported(tmp_path):
    with pytest.raises(DatasetError, match="file not found"):
        load_dataset(tmp_path / "nowhere")

    duplicated = write_dataset(tmp_path / "dup", [DOCUMENT, DOCUMENT], [question()])
    with pytest.raises(DatasetError, match="unique"):
        load_dataset(duplicated)

    broken = write_dataset(tmp_path / "broken", [DOCUMENT], [question()])
    (broken / "corpus.jsonl").write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="invalid JSON"):
        load_dataset(broken)

    empty = write_dataset(tmp_path / "empty", [DOCUMENT], [question()])
    (empty / "questions.jsonl").write_text("\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="at least one"):
        load_dataset(empty)
