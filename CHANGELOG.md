# Changelog

All notable changes to chunkaudit are listed here. Versions follow [semantic versioning](https://semver.org/).

## 0.3.0 - 2026-11-15

- `--show-splits N` lists the answers each strategy split, with `||` where a chunk ends inside the evidence.
- `split_examples()` and `SplitExample` are part of the library API.

## 0.2.0 - 2026-11-14

- The default output is now an aligned table that is easy to read in a terminal (`--format table`). Use `--format markdown` for the previous default.
- `render_table` is part of the library API.
- Demo recordings in the README, generated from real output by `scripts/make_demo_gif.py`.
- README: measured results on SQuAD v1.1 (48 documents, 10,570 questions) and an overlap sweep.

## 0.1.0 - 2026-11-14

- Four chunking strategies with character spans: `fixed`, `sentence`, `paragraph` and `recursive`.
- Audit metrics: answer split rate, Recall@k, MRR and words in the top k chunks.
- BM25 retriever on an inverted index; custom chunkers and retrievers can be plugged in.
- `chunkaudit run`, `chunkaudit demo` and `chunkaudit squad` commands, with Markdown, JSON and CSV output.
- A sample dataset of 5 articles and 20 questions, written for this project.
