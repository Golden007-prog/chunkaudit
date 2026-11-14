# Changelog

All notable changes to chunkaudit are listed here. Versions follow [semantic versioning](https://semver.org/).

## Unreleased

- README: measured results on SQuAD v1.1 (48 documents, 10,570 questions) and an overlap sweep.

## 0.1.0 - 2026-11-14

- Four chunking strategies with character spans: `fixed`, `sentence`, `paragraph` and `recursive`.
- Audit metrics: answer split rate, Recall@k, MRR and words in the top k chunks.
- BM25 retriever on an inverted index; custom chunkers and retrievers can be plugged in.
- `chunkaudit run`, `chunkaudit demo` and `chunkaudit squad` commands, with Markdown, JSON and CSV output.
- A sample dataset of 5 articles and 20 questions, written for this project.
