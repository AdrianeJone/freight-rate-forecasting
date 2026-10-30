"""FastAPI service that predicts the posted rate of a freight load.

Usage (from the repo root):
    uvicorn api:app --app-dir src
Then open http://127.0.0.1:8000/docs
"""
import os
from contextlib import asynccontextmanager
from datetime import date as Date
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from pipeline import make_features

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/freight_model.joblib"))
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run: python src/train_model.py --data-dir data"
        )
    state["model"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="Freight Rate Forecasting API", version="1.0", lifespan=lifespan)


class Load(BaseModel):
    pickup_lat: float = Field(..., ge=-90, le=90)
    pickup_lon: float = Field(..., ge=-180, le=180)
    delivery_lat: float = Field(..., ge=-90, le=90)
    delivery_lon: float = Field(..., ge=-180, le=180)
    distance: float = Field(..., gt=0, description="Miles")
    equipment: Literal["Dry Van", "Flatbed", "Reefer"]
    weight: Optional[float] = Field(None, description="Pounds. Leave empty if unknown; it is imputed.")
    date: Date
    market_index: Optional[float] = Field(None, description="Leave empty if unknown; it is imputed.")
    quote_signal: float


class Prediction(BaseModel):
    predicted_rate: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=Prediction)
def predict(load: Load):
    row = pd.DataFrame([load.model_dump()])
    row[["weight", "market_index"]] = row[["weight", "market_index"]].astype(float)  # None -> NaN
    rate = state["model"].predict(make_features(row))[0]
    return Prediction(predicted_rate=round(float(rate), 2))