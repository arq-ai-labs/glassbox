# Contributing to Glassbox

Thanks for your interest in contributing.

## Setup

```bash
git clone https://github.com/arq-ai-labs/glassbox.git
cd glassbox
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Quality

```bash
ruff check src/ tests/    # Linting
mypy src/glassbox --ignore-missing-imports  # Type checking
```

## Pull Requests

1. Fork the repo and create a branch from `main`.
2. Add tests for any new functionality.
3. Make sure all tests pass and code is lint-clean.
4. Open a PR with a clear description of what changed and why.

## ContextPack Format Changes

Changes to the ContextPack format (section types, fields, schema) go through the spec at `spec/CONTEXTPACK_SPEC.md`. Open an issue first to discuss before submitting format changes.
