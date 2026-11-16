import csv
import io
import json

import pytest

from chunkaudit import __version__, load_dataset, sample_path
from chunkaudit.cli import main
from chunkaudit.squad import convert_squad


def test_demo_prints_an_aligned_table_by_default(capsys):
    assert main(["demo"]) == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0].startswith("Strategy") and "Answer split" in lines[0]
    assert set(lines[1]) == {"-", " "}
    assert len(lines) == 2 + 7
    assert len({len(line) for line in lines}) == 1  # every row has the same width
    assert lines[0].index("Answer split") + len("Answer split") == lines[2].index("%") + 1
    assert "5 documents | 20 questions" in captured.err


def test_markdown_format_prints_a_markdown_table(capsys):
    assert main(["demo", "--format", "markdown"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("| Strategy | Chunks | Avg words | Answer split | Recall@1")
    assert lines[2].startswith("| `")
    assert len(lines) == 2 + 7


def test_run_with_options_and_json_output(capsys):
    code = main(
        [
            "run",
            str(sample_path()),
            "-s",
            "fixed:40, sentence:2",
            "-k",
            "1,2",
            "--mrr-depth",
            "5",
            "--format",
            "json",
            "--sort",
            "given",
        ]
    )
    assert code == 0
    records = json.loads(capsys.readouterr().out)
    assert [record["strategy"] for record in records] == ["fixed:40", "sentence:2"]
    assert set(records[0]["recall"]) == {"1", "2"}
    assert records[0]["mrr_depth"] == 5


def test_csv_output_and_sorting_by_split_rate(capsys):
    assert main(["demo", "--format", "csv", "--sort", "split"]) == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    rates = [float(row["split_rate"]) for row in rows]
    assert rates == sorted(rates)
    assert "recall@5" in rows[0] and "context_words@5" in rows[0]


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "no/such/folder"],
        ["demo", "-s", "semantic:10"],
        ["demo", "-s", ","],
        ["demo", "--mrr-depth", "0"],
    ],
)
def test_bad_input_exits_with_code_2_and_a_message(arguments, capsys):
    assert main(arguments) == 2
    assert capsys.readouterr().err.startswith("error: ")


def test_bad_cutoffs_and_version(capsys):
    with pytest.raises(SystemExit) as error:
        main(["demo", "-k", "0"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0
    assert __version__ in capsys.readouterr().out


SQUAD = {
    "data": [
        {
            "title": "Tides",
            "paragraphs": [
                {
                    "context": "Tides rise and fall. The Moon causes most of the effect.",
                    "qas": [
                        {
                            "id": "a1",
                            "question": "What causes most of the effect? ",
                            "answers": [{"text": "The Moon", "answer_start": 21}],
                        },
                        {
                            "id": "a2",
                            "question": "Wrong offset?",
                            "answers": [{"text": "The Moon", "answer_start": 3}],
                        },
                    ],
                },
                {
                    "context": "Spring tides are the largest. Neap tides are the smallest.",
                    "qas": [
                        {
                            "id": "a3",
                            "question": "Which tides are the smallest?",
                            "answers": [{"text": "Neap tides", "answer_start": 30}],
                        },
                        {"id": "a4", "question": "No answer?", "answers": []},
                    ],
                },
            ],
        },
        {"title": "Skipped", "paragraphs": []},
    ]
}


def test_squad_conversion_joins_paragraphs_and_shifts_offsets(tmp_path):
    source = tmp_path / "squad.json"
    source.write_text(json.dumps(SQUAD), encoding="utf-8")

    assert convert_squad(source, tmp_path / "sentence", max_docs=1) == (1, 2)
    dataset = load_dataset(tmp_path / "sentence")
    text = dataset.documents[0].text
    assert text.count("\n\n") == 1
    evidence = {item.id: text[item.start : item.end] for item in dataset.questions}
    assert evidence == {
        "a1": "The Moon causes most of the effect.",
        "a3": "Neap tides are the smallest.",
    }
    assert dataset.questions[0].text == "What causes most of the effect?"

    assert convert_squad(source, tmp_path / "answer", evidence="answer") == (2, 2)
    answers = load_dataset(tmp_path / "answer")
    spans = [answers.documents[0].text[item.start : item.end] for item in answers.questions]
    assert spans == ["The Moon", "Neap tides"]

    with pytest.raises(ValueError):
        convert_squad(source, tmp_path / "bad", evidence="paragraph")


def test_show_splits_prints_examples_after_the_table(capsys):
    assert main(["demo", "-s", "fixed:40,sentence:2", "--show-splits", "2"]) == 0
    out = capsys.readouterr().out
    assert "fixed:40: 12 of 20 answers split. Showing 2 (|| marks where a chunk ends):" in out
    assert out.count(" || ") >= 2
    assert "sentence:2: no answer was split." in out
    assert out.index("Strategy") < out.index("fixed:40: 12 of 20")


def test_show_splits_is_rejected_for_csv_and_negative_limits(capsys):
    assert main(["demo", "--show-splits", "1", "--format", "csv"]) == 2
    assert "--show-splits needs" in capsys.readouterr().err
    assert main(["demo", "--show-splits", "-1"]) == 2


@pytest.mark.parametrize("limit", [1, 2, 10])
def test_json_split_examples_are_structured_limited_and_follow_strategy_sort(
    tmp_path, capsys, limit
):
    text = "Alpha beta gamma delta epsilon zeta."
    (tmp_path / "corpus.jsonl").write_text(
        json.dumps({"id": "doc", "text": text}) + "\n", encoding="utf-8"
    )
    questions = [
        {"id": "q1", "question": "Which words?", "doc_id": "doc", "evidence": text},
        {
            "id": "q2",
            "question": "Middle words?",
            "doc_id": "doc",
            "evidence": "gamma delta epsilon",
        },
    ]
    (tmp_path / "questions.jsonl").write_text(
        "\n".join(json.dumps(q) for q in questions) + "\n", encoding="utf-8"
    )
    args = ["run", str(tmp_path), "-s", "fixed:2,sentence:1", "--format", "json", "--sort", "split"]
    assert main(args) == 0
    original = json.loads(capsys.readouterr().out)
    assert all("split_examples" not in record for record in original)

    assert main([*args, "--show-splits", str(limit)]) == 0
    records = json.loads(capsys.readouterr().out)
    assert [r["strategy"] for r in records] == ["sentence:1", "fixed:2"]
    assert records[0]["split_examples"] == []
    expected = [
        {
            "question_id": "q1",
            "question": "Which words?",
            "doc_id": "doc",
            "evidence": "Alpha beta || gamma delta || epsilon zeta.",
            "pieces": 3,
        },
        {
            "question_id": "q2",
            "question": "Middle words?",
            "doc_id": "doc",
            "evidence": "gamma delta || epsilon",
            "pieces": 2,
        },
    ]
    assert records[1]["split_examples"] == expected[:limit]
    assert all(record["seconds"] >= 0 for record in records)

    def stable(r):
        return {k: v for k, v in r.items() if k not in ("split_examples", "seconds")}

    assert [stable(record) for record in records] == [stable(record) for record in original]
