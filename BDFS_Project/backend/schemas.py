"""Pydantic request/response models for the BDFS API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FeaturesIn(BaseModel):
    session_load: float = Field(..., ge=0, le=2)
    rt_mean: float = Field(..., ge=0)
    rt_std: float = Field(..., ge=0)
    rt_slope: float
    rt_skew: float
    rt_kurtosis: float
    pref_reversal_rate: float = Field(..., ge=0, le=1)
    rolling_incon_5: float = Field(..., ge=0, le=1)
    rolling_incon_10: float = Field(..., ge=0, le=1)
    choice_entropy: float = Field(..., ge=0, le=1)
    session_position: float = Field(..., ge=0, le=1)
    fatigue_slope: float
    accuracy_decay_rate: float
    inter_trial_variability: float = Field(..., ge=0)
    ez_drift_rate: float = Field(..., ge=0)
    ez_boundary: float = Field(..., ge=0)
    ez_nondecision: float = Field(..., ge=0)
    drift_boundary_ratio: float = Field(..., ge=0)
    task_complexity_level: int = Field(..., ge=0, le=2)


class PredictRequest(BaseModel):
    features: FeaturesIn


class FeatureContribution(BaseModel):
    feature: str
    value: float


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_used: str
    predicted_class: int
    label: Literal["fatigued", "not_fatigued"]
    probability: float
    probability_not_fatigued: float
    top_contributing_features: list[FeatureContribution]


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]


class SchemaField(BaseModel):
    name: str
    type: str
    min: float
    max: float
    example: float
    group: str


class SchemaResponse(BaseModel):
    fields: list[SchemaField]
