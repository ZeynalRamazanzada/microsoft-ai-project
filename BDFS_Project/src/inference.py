"""
BDFS — Inference Service
========================
Wraps scaler + XGBoost model in a single predict() call.
Mirrors the training-time preprocessing exactly:
  raw dict -> one-hot complexity -> StandardScaler -> derive 6 features -> predict_proba
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src.feature_engineer import derive_new_features  # noqa: E402


RAW_NUMERIC_FEATURES: list[str] = [
    "session_load",
    "rt_mean",
    "rt_std",
    "rt_slope",
    "rt_skew",
    "rt_kurtosis",
    "pref_reversal_rate",
    "rolling_incon_5",
    "rolling_incon_10",
    "choice_entropy",
    "session_position",
    "fatigue_slope",
    "accuracy_decay_rate",
    "inter_trial_variability",
    "ez_drift_rate",
    "ez_boundary",
    "ez_nondecision",
    "drift_boundary_ratio",
]

COMPLEXITY_FIELD = "task_complexity_level"

SCALED_COLUMN_ORDER: list[str] = RAW_NUMERIC_FEATURES + [
    "complexity_complexity_1",
    "complexity_complexity_2",
]

INPUT_FIELDS: list[str] = RAW_NUMERIC_FEATURES + [COMPLEXITY_FIELD]

FEATURE_RANGES: dict[str, dict[str, float]] = {
    "session_load":            {"min": 0,     "max": 2,    "example": 1.0},
    "rt_mean":                 {"min": 0.0,   "max": 3.0,  "example": 0.75},
    "rt_std":                  {"min": 0.0,   "max": 1.5,  "example": 0.18},
    "rt_slope":                {"min": -0.5,  "max": 0.5,  "example": 0.003},
    "rt_skew":                 {"min": -3.0,  "max": 5.0,  "example": 0.4},
    "rt_kurtosis":             {"min": -3.0,  "max": 20.0, "example": 3.1},
    "pref_reversal_rate":      {"min": 0.0,   "max": 1.0,  "example": 0.55},
    "rolling_incon_5":         {"min": 0.0,   "max": 1.0,  "example": 0.42},
    "rolling_incon_10":        {"min": 0.0,   "max": 1.0,  "example": 0.40},
    "choice_entropy":          {"min": 0.0,   "max": 1.0,  "example": 0.68},
    "session_position":        {"min": 0.0,   "max": 1.0,  "example": 0.6},
    "fatigue_slope":           {"min": -0.5,  "max": 0.5,  "example": 0.01},
    "accuracy_decay_rate":     {"min": -0.1,  "max": 0.1,  "example": 0.015},
    "inter_trial_variability": {"min": 0.0,   "max": 1.0,  "example": 0.12},
    "ez_drift_rate":           {"min": 0.05,  "max": 1.0,  "example": 0.35},
    "ez_boundary":             {"min": 0.5,   "max": 2.0,  "example": 1.4},
    "ez_nondecision":          {"min": 0.05,  "max": 0.5,  "example": 0.18},
    "drift_boundary_ratio":    {"min": 0.0,   "max": 2.0,  "example": 0.25},
    "task_complexity_level":   {"min": 0,     "max": 2,    "example": 1},
}


def _one_hot_complexity(level: int) -> tuple[int, int]:
    """Match preprocessor.py: pd.get_dummies(drop_first=True) on values {0,1,2}."""
    level = int(round(level))
    level = max(0, min(2, level))
    return (1 if level == 1 else 0, 1 if level == 2 else 0)


class BDFSPredictor:
    """Singleton predictor. Load once at process startup."""

    _instance: "BDFSPredictor | None" = None

    def __init__(self) -> None:
        splits_path = PROJECT_ROOT / config.SPLIT_INDICES_PATH
        model_path = PROJECT_ROOT / config.MODELS_DIR / "xgb_best.pkl"
        shap_path = PROJECT_ROOT / config.RESULTS_DIR / "shap_feature_importance.csv"

        if not splits_path.exists():
            raise FileNotFoundError(f"Missing splits pkl: {splits_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")

        splits = joblib.load(splits_path)
        self.scaler = splits["scaler"]
        self.feature_names_full: list[str] = list(splits["feature_names"])
        del splits  # free memory; we don't need X/y arrays

        self.model = joblib.load(model_path)
        self.model_name = "xgb"

        if shap_path.exists():
            shap_df = pd.read_csv(shap_path)
            self.shap_ranking: list[str] = shap_df["Feature"].tolist()
        else:
            self.shap_ranking = []

    @classmethod
    def get(cls) -> "BDFSPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _validate(self, raw: dict[str, Any]) -> dict[str, float]:
        missing = [f for f in INPUT_FIELDS if f not in raw]
        if missing:
            raise ValueError(f"Missing fields: {missing}")
        clean: dict[str, float] = {}
        for f in INPUT_FIELDS:
            try:
                clean[f] = float(raw[f])
            except (TypeError, ValueError) as e:
                raise ValueError(f"Field '{f}' must be numeric, got {raw[f]!r}") from e
        return clean

    def _build_row(self, clean: dict[str, float]) -> pd.DataFrame:
        c1, c2 = _one_hot_complexity(clean[COMPLEXITY_FIELD])
        row: dict[str, float] = {f: clean[f] for f in RAW_NUMERIC_FEATURES}
        row["complexity_complexity_1"] = c1
        row["complexity_complexity_2"] = c2
        return pd.DataFrame([row], columns=SCALED_COLUMN_ORDER)

    def predict(self, raw: dict[str, Any]) -> dict[str, Any]:
        clean = self._validate(raw)
        df_raw = self._build_row(clean)

        scaled = self.scaler.transform(df_raw.values)

        engineered, full_names = derive_new_features(scaled, SCALED_COLUMN_ORDER)
        engineered = np.nan_to_num(engineered, nan=0.0, posinf=0.0, neginf=0.0)

        proba = self.model.predict_proba(engineered)[0]
        prob_fatigued = float(proba[1])
        prob_not = float(proba[0])
        predicted_class = int(prob_fatigued >= 0.5)

        top_contributors = self._top_contributors(engineered[0], full_names)

        return {
            "model_used": self.model_name,
            "predicted_class": predicted_class,
            "label": "fatigued" if predicted_class == 1 else "not_fatigued",
            "probability": round(prob_fatigued, 4),
            "probability_not_fatigued": round(prob_not, 4),
            "top_contributing_features": top_contributors,
        }

    def _top_contributors(
        self, engineered_row: np.ndarray, full_names: list[str], k: int = 3
    ) -> list[dict[str, Any]]:
        if not self.shap_ranking:
            return []
        name_to_idx = {n: i for i, n in enumerate(full_names)}
        out: list[dict[str, Any]] = []
        for feat in self.shap_ranking:
            if feat in name_to_idx:
                out.append(
                    {"feature": feat, "value": round(float(engineered_row[name_to_idx[feat]]), 4)}
                )
            if len(out) >= k:
                break
        return out


def get_predictor() -> BDFSPredictor:
    return BDFSPredictor.get()
