# Contributing

Thanks for taking a look. Bug reports, new chunking strategies and datasets are all welcome.

## Set up

```
git clone https://github.com/Vishal-Krishna-Kumar/chunkaudit.git
cd chunkaudit
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before you open a pull request

```
ruff check .
ruff format --check .
mypy chunkaudit
pytest
```

CI runs the same four commands on Python 3.10, 3.12 and 3.13.

## Adding a chunking strategy

1. Add a frozen dataclass to `chunkaudit/chunkers.py` with a `name` property and a `split(text)` method that returns `(start, end)` character spans.
2. Register it in `_KINDS` so it can be written as a spec such as `mykind:200`.
3. Add it to the parametrised tests in `tests/test_chunkers.py`. They check that every word of the sample documents lands in a chunk and that spans are valid.

Spans must point into the original text. Do not return modified text: the audit compares spans with the evidence positions.

## Reporting a bug

Use the issue template. The smallest document and question that reproduce the problem help the most.

## Releasing (maintainer)

1. Bump `__version__` in `chunkaudit/__init__.py` and add a section to `CHANGELOG.md`.
2. Merge, then publish a GitHub release tagged `vX.Y.Z`.
3. The `Release` workflow builds the package and uploads it to PyPI through trusted publishing. This needs a one-time trusted-publisher entry for this repository on PyPI.
