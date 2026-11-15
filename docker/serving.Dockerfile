# Serving image (Phase 6, Step 33): the serving app only. It installs the serving requirement set,
# which contains neither MLflow nor Pandera (the serving app must not import them).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AUTOML_IN_CONTAINER=1 \
    AUTOML_MODELS_DIR=/data/models

WORKDIR /app
COPY docker/requirements-serving.txt /tmp/requirements.txt
RUN pip install --require-hashes --no-deps -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY scripts/serve_model.py /app/scripts/serve_model.py
RUN pip install --no-deps -e /app

RUN useradd --system --uid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin automl \
    && mkdir -p /data/models && chown -R automl:automl /data
USER automl

CMD ["python", "scripts/serve_model.py", "--host", "0.0.0.0"]
