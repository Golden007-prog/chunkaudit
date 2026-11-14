import csv
import io
import json

import pytest

from chunkaudit import __version__, load_dataset, sample_path
from chunkaudit.cli import main
from chunkaudit.squad import convert_squad


def test_demo_prints_a_markdown_table(capsys):
    assert main(["demo"]) == 0
    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    assert lines[0].startswith("| Strategy | Chunks | Avg words | Answer split | Recall@1")
    assert len(lines) == 2 + 7
    assert "5 documents | 20 questions" in captured.err


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
