import json
import logging
import math
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import joblib
import redis
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model
# 1. Environment configuration
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
GIT_SHA = os.getenv("GIT_SHA", "unknown")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
redis_client = (
    redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=1,
    )
    if os.getenv("REDIS_HOST")
    else None
)

REDIS_COUNTER_KEY = "prediction_requests"
DECISION_THRESHOLD = float(
    os.getenv("DECISION_THRESHOLD", "0.5")
)
if not 0.0 <= DECISION_THRESHOLD <= 1.0:
    raise ValueError(
        "DECISION_THRESHOLD must be between 0 and 1"
    )

# 2. Logging to stdout
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger("mlops-api")
# 3. Model state
model: Any = None
model_card: dict = {}
PredictRequest: type[BaseModel]
prediction_count = 0
# 4. Read feature names from model card
MODEL_CARD_PATH = MODEL_PATH.with_name("model_card.json")
if not MODEL_CARD_PATH.is_file():
    raise FileNotFoundError(
        f"Model card not found: {MODEL_CARD_PATH}"
    )
with MODEL_CARD_PATH.open(encoding="utf-8") as file:
    model_card = json.load(file)
feature_names = model_card["feature_names"]
if len(feature_names) != 30:
    raise ValueError(
        "Model card must contain exactly 30 features"
    )
# 5. Strict Pydantic request validation
PredictRequest = create_model(
    "PredictRequest",
    __config__=ConfigDict(extra="forbid"),
    **{
        feature: (
            float,
            Field(ge=0, allow_inf_nan=False),
        )
        for feature in feature_names
    },
)
# 6. Load model once at application startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}"
        )

    logger.info("Loading model from %s", MODEL_PATH)

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "Loaded model does not support predict_proba"
        )

    if list(model.classes_) != [0, 1]:
        raise RuntimeError(
            "Expected model classes [0, 1]"
        )

    logger.info(
        "Model loaded successfully: %s",
        model_card["model_family"],
    )

    yield

    logger.info("Application shutting down")
# 7. Initialize FastAPI
app = FastAPI(
    title="Meridian Diagnostics Breast Cancer API",
    description=(
        "MLOps Assignment 2: Breast cancer "
        "classification inference service."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)
# 8. GET /health
@app.get("/health")
def health():
    return {"status": "ok"}
# 9. POST /predict
@app.post("/predict")
def predict(payload: PredictRequest):
    global prediction_count

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    # Preserve exact training feature order.
    values = payload.model_dump()

    ordered_values = {
        feature: values[feature]
        for feature in feature_names
    }

    input_df = pd.DataFrame(
        [ordered_values],
        columns=feature_names,
    )

    malignant_index = list(model.classes_).index(0)

    malignant_probability = float(
        model.predict_proba(input_df)[0][malignant_index]
    )

    label = (
        "malignant"
        if malignant_probability >= DECISION_THRESHOLD
        else "benign"
    )

    prediction_count += 1

    if redis_client is not None:
        try:
            redis_count = redis_client.incr(REDIS_COUNTER_KEY)
            logger.info("Redis prediction counter: %s", redis_count)
        except redis.exceptions.RedisError as error:
            logger.warning("Redis unavailable; prediction continues: %s", error)

    logger.info(
        "Prediction completed | label=%s | probability=%.4f",
        label,
        malignant_probability,
    )

    return {
        "label": label,
        "malignant_probability": round(
            malignant_probability, 4
        ),
        "threshold": DECISION_THRESHOLD,
        "model_version": MODEL_VERSION,
    }
# 10. GET /model/info
@app.get("/model/info")
def model_info():
    return {
        "model_version": MODEL_VERSION,
        "model_family": model_card["model_family"],
        "sklearn_version": model_card["sklearn_version"],
        "git_commit": GIT_SHA,
        "threshold": DECISION_THRESHOLD,
        "feature_count": model_card["feature_count"],
        "metrics": model_card["metrics"],
    }
#11
@app.get("/stats")
def stats():
    if redis_client is not None:
        try:
            count = redis_client.get(REDIS_COUNTER_KEY)
            return {
                "prediction_count": int(count or 0),
                "source": "redis",
                "redis_host": REDIS_HOST,
                "redis_port": REDIS_PORT,
            }
        except redis.exceptions.RedisError as error:
            logger.warning("Redis unavailable while reading stats: %s", error)

            return {
                "prediction_count": prediction_count,
                "source": "in_memory_fallback",
                "redis_available": False,
            }

    return {
        "prediction_count": prediction_count,
        "source": "in_memory",
        "redis_host": REDIS_HOST,
        "redis_port": REDIS_PORT,
    }
