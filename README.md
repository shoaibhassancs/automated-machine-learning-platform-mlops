# AutoML Ops

A single-user, local platform that manages the machine learning lifecycle for tabular CSV data:
dataset validation, training, experiment tracking, a model registry, deployment and monitoring.
Solo portfolio project, work in progress. Scope: see [SCOPE.md](SCOPE.md).

## Status

Setup only. There is no application code yet.

## Development setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                      # creates .venv and installs the dev tools
source .venv/bin/activate    # optional: activate the environment in a terminal
make check                   # lint, type-check and test
```
