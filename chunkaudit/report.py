"""Render audit results as a Markdown table, JSON or CSV."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict

from .evaluate import StrategyResult

SORT_KEYS: dict[str, Callable[[StrategyResult], float]] = {
    "mrr": lambda result: -result.mrr,
    "recall": lambda result: -result.recall[max(result.recall)],
    "split": lambda result: result.split_rate,
    "words": lambda result: result.context_words[max(result.context_words)],
}


def sort_results(results: Sequence[StrategyResult], key: str) -> list[StrategyResult]:
    """Order results by ``key`` (best first). ``given`` keeps the original order."""
    if key == "given":
        return list(results)
    return sorted(results, key=SORT_KEYS[key])


def render_markdown(results: Sequence[StrategyResult], mrr_depth: int = 10) -> str:
    if not results:
        return ""
    cutoffs = sorted(results[0].recall)
    largest = cutoffs[-1]
    header = ["Strategy", "Chunks", "Avg words", "Answer split"]
    header += [f"Recall@{k}" for k in cutoffs]
    header += [f"MRR@{mrr_depth}", f"Words in top {largest}"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|---|" + "---:|" * (len(header) - 1),
    ]
    for result in results:
        cells = [
            f"`{result.strategy}`",
            f"{result.chunks:,}",
            f"{result.avg_words:.0f}",
            f"{result.split_rate:.1%}",
            *(f"{result.recall[k]:.1%}" for k in cutoffs),
            f"{result.mrr:.3f}",
            f"{result.context_words[largest]:,.0f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_json(results: Sequence[StrategyResult], mrr_depth: int = 10) -> str:
    records = [{**asdict(result), "mrr_depth": mrr_depth} for result in results]
    return json.dumps(records, indent=2)


def render_csv(results: Sequence[StrategyResult], mrr_depth: int = 10) -> str:
    if not results:
        return ""
    cutoffs = sorted(results[0].recall)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        ["strategy", "chunks", "avg_words", "split_rate"]
        + [f"recall@{k}" for k in cutoffs]
        + [f"mrr@{mrr_depth}"]
        + [f"context_words@{k}" for k in cutoffs]
    )
    for result in results:
        writer.writerow(
            [
                result.strategy,
                result.chunks,
                round(result.avg_words, 1),
                round(result.split_rate, 4),
            ]
            + [round(result.recall[k], 4) for k in cutoffs]
            + [round(result.mrr, 4)]
            + [round(result.context_words[k], 1) for k in cutoffs]
        )
    return buffer.getvalue().rstrip("\n")


RENDERERS: dict[str, Callable[[Sequence[StrategyResult], int], str]] = {
    "markdown": render_markdown,
    "json": render_json,
    "csv": render_csv,
}
