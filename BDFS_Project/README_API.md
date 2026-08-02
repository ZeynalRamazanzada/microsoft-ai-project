# BDFS — Web App (API + UI)

A FastAPI backend serving the trained XGBoost model + a React/Vite frontend
for live single-trial fatigue prediction.

```
BDFS_Project/
├── backend/          FastAPI app  (port 8000)
├── frontend/         Vite + React (port 5173)
├── src/inference.py  BDFSPredictor (loads scaler + model)
└── models/xgb_best.pkl
```

## One-time setup

> The original `.venv` had a broken `site` config (UnicodeDecodeError on a
> `.pth` file). The API uses a fresh venv at `.venv_api/`.

### Backend

```bash
cd "BDFS_Project"
/opt/homebrew/bin/python3.11 -m venv .venv_api
.venv_api/bin/python -m ensurepip --upgrade
.venv_api/bin/python -m pip install -r backend/requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

## Running (two terminals)

**Terminal A — backend**

```bash
cd "BDFS_Project"
.venv_api/bin/uvicorn backend.main:app --reload --port 8000
```

OpenAPI docs: <http://localhost:8000/docs>

**Terminal B — frontend**

```bash
cd "BDFS_Project/frontend"
npm run dev
```

Open <http://localhost:5173>.

The Vite dev server proxies `/api/*` → `http://localhost:8000/*`, so the UI
talks to the backend with no CORS issues during development.

## API reference

| Method | Path                       | Purpose                                      |
| ------ | -------------------------- | -------------------------------------------- |
| GET    | `/health`                  | Liveness + which models are loaded           |
| GET    | `/schema`                  | Field list + ranges (drives the UI form)     |
| GET    | `/example/{fatigued\|not_fatigued}` | Returns a random raw row matching label |
| POST   | `/predict`                 | Single-trial prediction                      |
| POST   | `/predict/batch`           | CSV upload → CSV with predictions            |

### Example: curl

```bash
curl -s http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "features": {
      "session_load": 1, "rt_mean": 0.72, "rt_std": 0.15,
      "rt_slope": 0.002, "rt_skew": 0.4, "rt_kurtosis": 3.1,
      "pref_reversal_rate": 0.55, "rolling_incon_5": 0.42,
      "rolling_incon_10": 0.40, "choice_entropy": 0.68,
      "session_position": 0.6, "fatigue_slope": 0.01,
      "accuracy_decay_rate": 0.015, "inter_trial_variability": 0.12,
      "ez_drift_rate": 0.35, "ez_boundary": 1.4, "ez_nondecision": 0.18,
      "drift_boundary_ratio": 0.25, "task_complexity_level": 2
    }
  }'
```

## How a prediction is computed

1. Validate the 19 raw inputs (Pydantic).
2. One-hot encode `task_complexity_level` → `complexity_complexity_1`, `_2`
   (matches `pd.get_dummies(drop_first=True)` used in training).
3. `scaler.transform()` on the 20-column vector using the persisted
   `StandardScaler` from `data/splits/split_indices.pkl`.
4. `derive_new_features()` from `src/feature_engineer.py` adds 6 engineered
   features → 26-dim vector.
5. Replace any NaN/Inf with 0 (same as training).
6. `xgb_best.predict_proba()` → probability of fatigue.
7. Top contributors = top features from `results/shap_feature_importance.csv`
   (precomputed global importance — no live SHAP call per request).

## Files added by this task

- `src/inference.py`
- `backend/__init__.py`, `backend/main.py`, `backend/schemas.py`, `backend/requirements.txt`
- `frontend/` (Vite + React + TS scaffold)
- `README_API.md` (this file)

Nothing under `models/`, `data/`, `results/`, or the existing training
scripts was modified. Your `app.py` Streamlit dashboard still works
unchanged.
