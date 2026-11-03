# Stage 1: Train the model
FROM python:3.12.12-slim-bookworm AS trainer

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements-train.txt .
RUN pip install --timeout 180 --retries 10 -r requirements-train.txt

COPY training/ ./training/
COPY params.yaml ./
COPY data/raw/breast_cancer.csv ./data/raw/

ARG MODEL_VERSION=1.0.0

RUN python training/train.py \
    --data-path data/raw/breast_cancer.csv \
    --output-dir artifacts \
    --model-version "${MODEL_VERSION}"


# Stage 2: Serve the trained model
FROM python:3.12.12-slim-bookworm AS production

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/artifacts/model.joblib \
    DECISION_THRESHOLD=0.5 \
    LOG_LEVEL=INFO \
    REDIS_HOST=localhost \
    REDIS_PORT=6379

COPY requirements-serve.txt .
RUN pip install --timeout 180 --retries 10 -r requirements-serve.txt

COPY app/ ./app/
COPY --from=trainer /build/artifacts/ ./artifacts/

ARG MODEL_VERSION=1.0.0
ARG GIT_SHA=unknown

ENV MODEL_VERSION=${MODEL_VERSION} \
    GIT_SHA=${GIT_SHA}

LABEL org.opencontainers.image.version="${MODEL_VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}"

RUN useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

ENTRYPOINT ["uvicorn", "app.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
