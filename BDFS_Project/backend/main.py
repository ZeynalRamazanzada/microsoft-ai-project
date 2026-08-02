"""
BDFS — FastAPI Backend
======================
Run from the project root (BDFS_Project/):
    .venv_api/bin/uvicorn backend.main:app --reload --port 8000

Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Path as PathParam, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from backend.schemas import (  # noqa: E402
    HealthResponse,
    PredictRequest,
    PredictResponse,
    SchemaField,
    SchemaResponse,
)
from src.inference import (  # noqa: E402
    FEATURE_RANGES,
    INPUT_FIELDS,
    BDFSPredictor,
    get_predictor,
)

FIELD_GROUPS: dict[str, str] = {
    "rt_mean": "Reaction Time",
    "rt_std": "Reaction Time",
    "rt_slope": "Reaction Time",
    "rt_skew": "Reaction Time",
    "rt_kurtosis": "Reaction Time",
    "inter_trial_variability": "Reaction Time",
    "pref_reversal_rate": "Choice Behavior",
    "rolling_incon_5": "Choice Behavior",
    "rolling_incon_10": "Choice Behavior",
    "choice_entropy": "Choice Behavior",
    "ez_drift_rate": "DDM Parameters",
    "ez_boundary": "DDM Parameters",
    "ez_nondecision": "DDM Parameters",
    "drift_boundary_ratio": "DDM Parameters",
    "session_load": "Session / Context",
    "session_position": "Session / Context",
    "fatigue_slope": "Session / Context",
    "accuracy_decay_rate": "Session / Context",
    "task_complexity_level": "Session / Context",
}


app = FastAPI(
    title="BDFS Fatigue Detection API",
    description="Predict behavioral decision fatigue from a single trial's features.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "null",                       # file:// origin
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warmup() -> None:
    BDFSPredictor.get()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    predictor = get_predictor()
    return HealthResponse(status="ok", models_loaded=[predictor.model_name])


@app.get("/schema", response_model=SchemaResponse)
def schema() -> SchemaResponse:
    fields: list[SchemaField] = []
    for name in INPUT_FIELDS:
        meta = FEATURE_RANGES[name]
        field_type = "int" if name == "task_complexity_level" else "float"
        fields.append(
            SchemaField(
                name=name,
                type=field_type,
                min=meta["min"],
                max=meta["max"],
                example=meta["example"],
                group=FIELD_GROUPS.get(name, "Other"),
            )
        )
    return SchemaResponse(fields=fields)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    predictor = get_predictor()
    try:
        result = predictor.predict(req.features.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PredictResponse(**result)


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)) -> StreamingResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    missing = [c for c in INPUT_FIELDS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV missing required columns: {missing}",
        )

    predictor = get_predictor()
    preds: list[int] = []
    probs: list[float] = []
    for _, row in df[INPUT_FIELDS].iterrows():
        result = predictor.predict(row.to_dict())
        preds.append(result["predicted_class"])
        probs.append(result["probability"])

    df_out = df.copy()
    df_out["predicted_class"] = preds
    df_out["probability_fatigued"] = probs

    buf = io.StringIO()
    df_out.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="bdfs_predictions.csv"'},
    )


@app.get("/example/{label}")
def example(
    label: str = PathParam(..., pattern="^(fatigued|not_fatigued)$"),
) -> dict[str, float | int]:
    target_class = 1 if label == "fatigued" else 0
    raw_path = PROJECT_ROOT / "data" / "raw" / "bdfs_raw.csv"
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail=f"Raw data not found at {raw_path}")

    df = pd.read_csv(raw_path, usecols=INPUT_FIELDS + ["fatigue_class"])
    df = df[df["fatigue_class"] == target_class].dropna(subset=INPUT_FIELDS)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No rows with label={label}")

    row = df.sample(n=1, random_state=None).iloc[0]
    out: dict[str, float | int] = {}
    for field in INPUT_FIELDS:
        val = row[field]
        if field == "task_complexity_level":
            out[field] = int(round(float(val)))
        else:
            out[field] = float(val)
    return out
