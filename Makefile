.PHONY: lint test typecheck check

lint:
	uv run ruff check .

test:
	uv run pytest

typecheck:
	uv run mypy

check: lint typecheck test

# One-time demo-data preparation (Step 2). Downloads from archive.ics.uci.edu only if
# data/raw/ lacks the files, then writes data/demo/. Re-run safely; output is reproducible.
.PHONY: demo-data
demo-data:
	uv run python scripts/prepare_demo_data.py
