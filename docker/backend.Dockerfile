# Backend image (Phase 6, Step 33): the control-plane API, the job worker and the database tools.
# One image, three commands (see compose.full.yaml). Local use only; not published anywhere.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AUTOML_IN_CONTAINER=1 \
    AUTOML_STORE_DIR=/data/store \
    AUTOML_MODELS_DIR=/data/models \
    AUTOML_JOBS_MLFLOW_DIR=/data/mlflow \
    AUTOML_MLFLOW_BACKEND=sqlite

WORKDIR /app
COPY docker/requirements-backend.txt /tmp/requirements.txt
RUN pip install --require-hashes --no-deps -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY scripts /app/scripts
# The project itself, without its dependencies (already installed and hash-checked above).
RUN pip install --no-deps -e /app

# A fixed non-root user; the data volume is created owned by it.
RUN useradd --system --uid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin automl \
    && mkdir -p /data/store /data/models /data/mlflow \
    && chown -R automl:automl /data
USER automl
VOLUME ["/data"]

# The default command is the API; Compose overrides it for the worker and the migration step.
CMD ["python", "scripts/serve_control.py", "--host", "0.0.0.0"]
