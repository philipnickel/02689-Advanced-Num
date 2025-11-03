# 02689-Advanced-Num

Advanced Numerical Algorithms - DTU Course 02689

## Setup

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Sync all dependencies (includes runtime, docs, and dev tools):
```bash
uv sync
```

## Usage

Run exercises and build documentation:
```bash
# Run all exercises
uv run python main.py --compute --plot --copy

# Build documentation
uv run python main.py --build-docs

# Clean documentation
uv run python main.py --clean-docs
```

## Dependencies
Working LaTeX installation for the figures.

