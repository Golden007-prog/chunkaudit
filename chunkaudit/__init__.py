"""chunkaudit: audit RAG chunking strategies without an API key."""

__version__ = "0.1.0"

from .bm25 import BM25Index
from .chunkers import (
    DEFAULT_STRATEGIES,
    Chunker,
    FixedChunker,
    ParagraphChunker,
    RecursiveChunker,
    SentenceChunker,
    parse_strategy,
)
from .dataset import Dataset, DatasetError, Document, Question, load_dataset, sample_path
from .evaluate import StrategyResult, compare, evaluate
from .report import render_csv, render_json, render_markdown, sort_results

__all__ = [
    "DEFAULT_STRATEGIES",
    "BM25Index",
    "Chunker",
    "Dataset",
    "DatasetError",
    "Document",
    "FixedChunker",
    "ParagraphChunker",
    "Question",
    "RecursiveChunker",
    "SentenceChunker",
    "StrategyResult",
    "__version__",
    "compare",
    "evaluate",
    "load_dataset",
    "parse_strategy",
    "render_csv",
    "render_json",
    "render_markdown",
    "sample_path",
    "sort_results",
]
