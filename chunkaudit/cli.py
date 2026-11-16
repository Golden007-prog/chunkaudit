"""Command line interface.

chunkaudit demo                         audit the bundled sample dataset
chunkaudit run DATASET                  audit your own dataset
chunkaudit squad --out data/squad       download SQuAD v1.1 and convert it
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .bm25 import BM25Index
from .chunkers import DEFAULT_STRATEGIES, Chunker, parse_strategy
from .dataset import Dataset, DatasetError, load_dataset, sample_path
from .embeddings import DEFAULT_MODEL, EmbeddingRetriever
from .evaluate import RetrieverFactory, compare, split_examples
from .report import RENDERERS, SORT_KEYS, render_json, sort_results
from .squad import convert_squad, download_squad

DEMO_STRATEGIES = (
    "fixed:40",
    "fixed:40:10",
    "fixed:80",
    "sentence:2",
    "sentence:3:1",
    "paragraph:80",
    "recursive:60",
)


def _numbers(value: str) -> list[int]:
    try:
        numbers = [int(part) for part in value.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("expected whole numbers separated by commas") from None
    if not numbers or min(numbers) < 1:
        raise argparse.ArgumentTypeError("expected positive whole numbers separated by commas")
    return numbers


def _add_audit_options(parser: argparse.ArgumentParser, strategies: Sequence[str]) -> None:
    parser.add_argument(
        "-s",
        "--strategies",
        default=",".join(strategies),
        help="comma-separated strategy specs (default: %(default)s)",
    )
    parser.add_argument(
        "-k", type=_numbers, default=[1, 3, 5], help="recall cutoffs (default: 1,3,5)"
    )
    parser.add_argument(
        "--mrr-depth", type=int, default=10, help="ranks considered for MRR (default: 10)"
    )
    parser.add_argument(
        "--format",
        choices=sorted(RENDERERS),
        default="table",
        help="table (default) is aligned for the terminal; the others are for pasting and scripts",
    )
    parser.add_argument(
        "--sort",
        choices=[*sorted(SORT_KEYS), "given"],
        default="mrr",
        help="order of the rows, best first (default: mrr)",
    )
    parser.add_argument(
        "--show-splits",
        type=int,
        default=0,
        metavar="N",
        help="show up to N marked split answers per strategy (table, markdown or json)",
    )
    parser.add_argument(
        "--retriever",
        choices=("bm25", "embeddings"),
        default="bm25",
        help='bm25 (default) needs nothing extra; embeddings needs "chunkaudit[embeddings]"',
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_MODEL,
        metavar="NAME",
        help="sentence-transformers model for --retriever embeddings (default: %(default)s)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chunkaudit",
        description="Audit RAG chunking strategies: how often they split the answer, "
        "and whether retrieval still finds it.",
    )
    parser.add_argument("--version", action="version", version=f"chunkaudit {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="audit a dataset folder")
    run.add_argument("dataset", type=Path, help="folder with corpus.jsonl and questions.jsonl")
    _add_audit_options(run, DEFAULT_STRATEGIES)

    demo = commands.add_parser("demo", help="audit the bundled sample dataset")
    _add_audit_options(demo, DEMO_STRATEGIES)

    squad = commands.add_parser("squad", help="download SQuAD v1.1 and convert it")
    squad.add_argument("--out", type=Path, default=Path("data/squad"))
    squad.add_argument("--cache", type=Path, default=Path(".chunkaudit_cache"))
    squad.add_argument("--evidence", choices=("sentence", "answer"), default="sentence")
    squad.add_argument("--max-docs", type=int, default=None, help="convert only the first N")
    return parser


def _audit(args: argparse.Namespace, dataset_path: Path) -> int:
    try:
        chunkers = [parse_strategy(spec) for spec in args.strategies.split(",") if spec.strip()]
        if not chunkers:
            raise ValueError("no strategies given")
        if args.mrr_depth < 1:
            raise ValueError("--mrr-depth must be at least 1")
        if args.show_splits < 0:
            raise ValueError("--show-splits must be 0 or more")
        if args.show_splits and args.format not in ("table", "markdown", "json"):
            raise ValueError("--show-splits needs --format table, markdown or json")
        dataset = load_dataset(dataset_path)
    except (DatasetError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    retriever: RetrieverFactory = BM25Index
    label = "BM25"
    if args.retriever == "embeddings":
        try:
            embedder = EmbeddingRetriever(args.embedding_model)
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        embedder.warm([question.text for question in dataset.questions])
        retriever = embedder
        label = f"embeddings ({args.embedding_model})"

    print(
        f"dataset: {dataset.name} | {len(dataset.documents)} documents | "
        f"{len(dataset.questions)} questions | retriever: {label}",
        file=sys.stderr,
    )
    results = compare(dataset, chunkers, args.k, args.mrr_depth, retriever)
    if args.show_splits and args.format == "json":
        examples = {
            chunker.name: split_examples(dataset, chunker, limit=args.show_splits)
            for chunker in chunkers
        }
        print(render_json(sort_results(results, args.sort), args.mrr_depth, examples=examples))
    else:
        print(RENDERERS[args.format](sort_results(results, args.sort), args.mrr_depth))
    if args.show_splits and args.format != "json":
        for chunker in chunkers:
            print()
            print(_render_splits(dataset, chunker, args.show_splits))
    return 0


def _render_splits(dataset: Dataset, chunker: Chunker, limit: int) -> str:
    """Examples of the answers a strategy cut, with ``||`` where a chunk ends."""
    examples = split_examples(dataset, chunker)
    if not examples:
        return f"{chunker.name}: no answer was split."
    shown = examples[:limit]
    lines = [
        f"{chunker.name}: {len(examples)} of {len(dataset.questions)} answers split. "
        f"Showing {len(shown)} (|| marks where a chunk ends):"
    ]
    for example in shown:
        lines.append(f"  [{example.question_id}] {example.question}")
        lines.append(f"      {example.evidence}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "squad":
        source = download_squad(args.cache)
        documents, questions = convert_squad(source, args.out, args.evidence, args.max_docs)
        print(f"wrote {documents} documents and {questions} questions to {args.out}")
        return 0
    return _audit(args, sample_path() if args.command == "demo" else args.dataset)


if __name__ == "__main__":
    raise SystemExit(main())
