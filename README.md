<div align="center">

# 🧠 BDFS — Behavioral Decision Fatigue Scoring

**Detecting and predicting human decision fatigue from single-trial behavioral and cognitive indicators using machine learning**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Best_Model-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

*An end-to-end ML pipeline that models hidden cognitive states, response time dynamics, preference reversals, and drift-diffusion parameters to classify whether a user is experiencing decision fatigue.*

**Authors:** Zeynalabdin Ramazanzada · Alperen Sümeroğlu

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset & Data Generation](#-dataset--data-generation)
- [ML Pipeline & Feature Engineering](#-ml-pipeline--feature-engineering)
- [Model Performance](#-model-performance)
- [SHAP Interpretability & Ablation Study](#-shap-interpretability--ablation-study)
- [Web Application & API](#-web-application--api)
- [Installation & Usage](#-installation--usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

**BDFS (Behavioral Decision Fatigue Scoring)** is a comprehensive machine learning system designed to detect and predict human decision fatigue from single-trial behavioral and cognitive indicators.

Decision fatigue refers to the deteriorating quality of decisions made by an individual after a long session of decision-making. BDFS models hidden cognitive states using:

- **Response time dynamics** (mean, variability, slope, distribution shape)
- **Preference reversal patterns** (short/long-term inconsistency)
- **EZ-Diffusion Model parameters** (drift rate, boundary separation, non-decision time)
- **Session-level fatigue trajectory** (position, load, accuracy decay)

The system classifies each trial as `fatigued` (1) or `not_fatigued` (0), achieving **0.9667 ROC-AUC** with the best model (XGBoost).

---

## ✨ Key Features

- 🔬 **End-to-End ML Pipeline** — From synthetic data generation to model evaluation in a single `run_all.py` command
- 🧬 **Theory-Driven Data Generation** — Ornstein-Uhlenbeck stochastic process + EZ-Diffusion Model for realistic cognitive modeling
- 🤖 **5-Model Comparison** — Logistic Regression, Random Forest, XGBoost, SVM, and KNN with 3-fold Stratified CV hyperparameter tuning
- 🎯 **SHAP Interpretability** — Full feature importance analysis with SHAP values for model transparency
- 📉 **Ablation Study** — Systematic feature group contribution analysis (Temporal vs DDM vs Baseline)
- 📊 **McNemar Statistical Test** — Statistically validated model improvement over baseline
- ⚡ **FastAPI REST Backend** — Production-ready API with single-trial and batch prediction endpoints
- 🖥️ **React + Vite Frontend** — Modern TypeScript UI with live prediction form and risk gauge
- 📈 **Streamlit Dashboard** — Interactive exploration of metrics, confusion matrices, ROC/PR curves, and SHAP plots
- 📄 **Publication-Quality Figures** — ROC curves, PR curves, confusion matrices, and SHAP summary visualizations

---

## 🏗️ Architecture

```
┌──────────────────────┐     ┌────────────────────┐     ┌─────────────────────┐
│  Data Generation     │────▶│  Preprocessing     │────▶│  Feature            │
│  (O-U Process +      │     │  (Imputation,      │     │  Engineering        │
│   DDM Parameters)    │     │   Scaling, Split)  │     │  (19 raw → 26 feat) │
└──────────────────────┘     └────────────────────┘     └──────────┬──────────┘
                                                                   │
                             ┌────────────────────┐                │
                             │  Model Training    │◀───────────────┘
                             │  (LR, RF, XGB,     │
                             │   SVM, KNN)        │
                             └────────┬───────────┘
                                      │
                    ┌─────────────────┼──────────────────┐
                    ▼                 ▼                   ▼
          ┌─────────────┐   ┌────────────────┐   ┌─────────────────┐
          │  Evaluation  │   │  SHAP Analysis │   │  Ablation Study │
          │  & McNemar   │   │  & Feature     │   │  (Feature Group │
          │  Test        │   │  Importance    │   │   Contribution) │
          └──────┬──────┘   └────────────────┘   └─────────────────┘
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
┌─────────────┐    ┌────────────────┐    ┌──────────────────┐
│  FastAPI     │    │  React + Vite  │    │  Streamlit       │
│  REST API   │◀──▶│  Frontend      │    │  Dashboard       │
│  (Port 8000)│    │  (Port 5173)   │    │  (Port 8501)     │
└─────────────┘    └────────────────┘    └──────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.11+, TypeScript |
| **ML / Data Science** | scikit-learn, XGBoost, imbalanced-learn (SMOTE), SciPy, statsmodels |
| **Interpretability** | SHAP |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | React 18, Vite, TypeScript |
| **Dashboard** | Streamlit |
| **Model Persistence** | joblib, pickle |

---

## 📊 Dataset & Data Generation

BDFS uses a **synthetically generated dataset** grounded in cognitive science theory:

| Parameter | Value |
|-----------|-------|
| **Total Records** | 150,000 trial-level records |
| **Participants** | 3,000 unique participants |
| **Trials per Participant** | 50 sequential decision trials |
| **Class Ratio** | ~35% fatigued / ~65% non-fatigued |
| **Missing Values** | 4% MCAR injection |
| **Outliers** | 3% (3–5x multiplied values) |

### Three-Tier Latent Variable Cascade

1. **L1 — Instantaneous Energy State:** Modeled via a discrete **Ornstein-Uhlenbeck (O-U)** stochastic process where energy depletes faster under high session load
2. **L2 — Session Trajectory:** Cumulative fatigue slope and position within the decision session
3. **L3 — Individual Resilience Threshold:** Participant baseline energy and fatigue resistance sampled from Beta distributions

### EZ-Diffusion Model (DDM) Parameters

- **Drift rate (v):** Speed of cognitive information processing
- **Boundary separation (a):** Decision caution threshold
- **Non-decision time (Ter):** Sensory/motor processing delay

---

## 🔬 ML Pipeline & Feature Engineering

### 19 Raw Observable Features

| Group | Features |
|-------|----------|
| **Reaction Time** | `rt_mean`, `rt_std`, `rt_slope`, `rt_skew`, `rt_kurtosis`, `inter_trial_variability` |
| **Choice Behavior** | `pref_reversal_rate`, `rolling_incon_5`, `rolling_incon_10`, `choice_entropy` |
| **Session/Context** | `session_load`, `session_position`, `fatigue_slope`, `accuracy_decay_rate`, `task_complexity_level` |
| **DDM Parameters** | `ez_drift_rate`, `ez_boundary`, `ez_nondecision`, `drift_boundary_ratio` |

### 6 Derived Features (→ 26 Total)

| Feature | Formula | Description |
|---------|---------|-------------|
| `rt_cv` | rt_std / (rt_mean + ε) | Reaction time variation coefficient |
| `incon_acceleration` | rolling_incon_5 − rolling_incon_10 | Short-term inconsistency acceleration |
| `ddm_fatigue_index` | ez_nondecision / (ez_drift_rate + ε) | Cognitive slowing index |
| `temporal_load_interaction` | session_position × session_load | Position-load interaction |
| `rt_kurtosis_normalized` | (rt_kurtosis − 3) / (rt_std + ε) | Normalized tail heaviness |
| `decision_efficiency` | (1 − pref_reversal_rate) / (rt_mean + ε) | Decision quality per unit time |

### Pipeline Steps

```
1. data_generator.py      → Generate 150K synthetic records
2. preprocessor.py        → Imputation, Winsorization, One-Hot, Stratified Split (70/15/15), Scaling
3. feature_engineer.py    → Correlation, VIF, Mutual Information, 6 derived features, RFE
4. train_step5.py         → Train 5 models with 3-fold Stratified K-Fold CV
5. evaluate_step6.py      → Test set evaluation (22,500 records) + McNemar test
6. ablation_step7.py      → Feature group ablation study
7. shap_step8.py          → SHAP value computation & visualization
```

---

## 📈 Model Performance

### Test Set Results (22,500 Hold-Out Records)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|:------|:--------:|:---------:|:------:|:--------:|:-------:|:------:|
| **XGBoost** ⭐ | **0.9053** | **0.8263** | **0.9094** | **0.8659** | **0.9667** | **0.9214** |
| Random Forest | 0.9044 | 0.8202 | 0.9163 | 0.8656 | 0.9672 | 0.9232 |
| Logistic Regression | 0.8863 | 0.7743 | 0.9339 | 0.8466 | 0.9476 | 0.8549 |
| KNN | 0.7029 | 0.5386 | 0.8107 | 0.6472 | 0.8123 | 0.6576 |
| SVM | 0.6708 | 0.5061 | 0.8650 | 0.6386 | 0.7713 | 0.5314 |

> **Best Model:** XGBoost — selected for optimal Precision-Recall balance, high ROC-AUC (0.9667), compact model size (537 KB vs RF's 99 MB), and rapid inference speed.

### XGBoost Hyperparameters

```
n_estimators=200, max_depth=5, learning_rate=0.1,
subsample=1.0, colsample_bytree=1.0, scale_pos_weight=1
```

### Statistical Validation

- **McNemar Test (XGBoost vs Logistic Regression):** χ² = 146.57, *p* = 9.76 × 10⁻³⁴ (*p* < 0.001)
- Confirms XGBoost provides a **statistically significant** improvement over the linear baseline

---

## 🔎 SHAP Interpretability & Ablation Study

### Top-10 Features by SHAP Importance

| Rank | Feature | Mean |SHAP| |
|:----:|---------|:------------:|
| 1 | `rolling_incon_5` | 2.5152 |
| 2 | `rolling_incon_10` | 1.1684 |
| 3 | `pref_reversal_rate` | 0.5230 |
| 4 | `ez_nondecision` | 0.3662 |
| 5 | `session_position` | 0.1275 |
| 6 | `incon_acceleration` | 0.0999 |
| 7 | `accuracy_decay_rate` | 0.0869 |
| 8 | `fatigue_slope` | 0.0784 |
| 9 | `rt_slope` | 0.0683 |
| 10 | `session_load` | 0.0666 |

### Ablation Study Results

| Feature Set | Impact on ROC-AUC | Impact on F1 |
|-------------|:-----------------:|:------------:|
| **Full (26 features)** | Baseline | 0.8659 |
| Without Temporal Features | **−0.0409** | 0.7846 |
| Without DDM Features | −0.0004 | ~0.865 |

> **Key Finding:** Temporal/dynamic behavioral features carry significantly more fatigue information than static DDM cognitive parameters.

---

## 🌐 Web Application & API

### FastAPI Backend (Port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service liveness & loaded model status |
| `GET` | `/schema` | Dynamic field descriptors, min/max bounds, and examples |
| `GET` | `/example/{label}` | Random historical sample (`fatigued` or `not_fatigued`) |
| `POST` | `/predict` | Single-trial prediction with probability & top SHAP contributors |
| `POST` | `/predict/batch` | Batch CSV upload → returns CSV with predictions |

#### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rt_mean": 0.85,
    "rt_std": 0.22,
    "rt_slope": 0.012,
    "rt_skew": 1.1,
    "rt_kurtosis": 4.2,
    "inter_trial_variability": 0.18,
    "pref_reversal_rate": 0.35,
    "rolling_incon_5": 0.45,
    "rolling_incon_10": 0.30,
    "choice_entropy": 0.92,
    "session_load": 7,
    "session_position": 42,
    "fatigue_slope": -0.015,
    "accuracy_decay_rate": 0.08,
    "task_complexity_level": 2,
    "ez_drift_rate": 0.12,
    "ez_boundary": 0.09,
    "ez_nondecision": 0.38,
    "drift_boundary_ratio": 1.33
  }'
```

#### Example Response

```json
{
  "prediction": 1,
  "label": "fatigued",
  "probability_fatigued": 0.874,
  "top_features": [
    {"feature": "rolling_incon_5", "shap_value": 2.31},
    {"feature": "rolling_incon_10", "shap_value": 1.05}
  ]
}
```

### React + Vite Frontend (Port 5173)

- Modern TypeScript + React interface with categorized input form (Reaction Time, Choice Behavior, DDM Parameters, Session/Context)
- One-click sample population buttons for `fatigued` / `not_fatigued` test cases
- Real-time risk percentage gauge and top SHAP feature contributors display

### Streamlit Dashboard

- Interactive exploration of model metrics, confusion matrices, ROC/PR curves, SHAP summary plots, and ablation study breakdown

---

## 🚀 Installation & Usage

### Prerequisites

- Python 3.11+
- Node.js 16+ (for the React frontend)

### 1. Clone the Repository

```bash
git clone https://github.com/microsoft-ai-project/BDFS_Fatigue_Scoring.git
cd BDFS_Fatigue_Scoring/BDFS_Project
```

### 2. Run the Complete ML Pipeline

```bash
python -m venv .venv_run
source .venv_run/bin/activate   # macOS / Linux
pip install -r requirements.txt

python run_all.py
```

> This sequentially executes: data generation → preprocessing → feature engineering → training → evaluation → ablation → SHAP analysis

### 3. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

### 4. Start the FastAPI Backend

```bash
python -m venv .venv_api
source .venv_api/bin/activate
pip install -r backend/requirements.txt

uvicorn backend.main:app --reload --port 8000
```

> 📖 Swagger docs available at `http://localhost:8000/docs`

### 5. Start the React Frontend

```bash
cd frontend
npm install
npm run dev
```

> 🌐 Web UI available at `http://localhost:5173`

---

## 📁 Project Structure

```
BDFS_Project/
│
├── src/                              # Core ML pipeline
│   ├── data_generator.py             # Synthetic data generation (O-U + DDM)
│   ├── preprocessor.py               # Imputation, scaling, stratified split
│   ├── feature_engineer.py           # Correlation, VIF, MI, derived features, RFE
│   ├── train_step5.py                # 5-model training with 3-fold CV
│   ├── evaluate_step6.py             # Test evaluation & McNemar test
│   ├── ablation_step7.py             # Feature group ablation study
│   ├── shap_step8.py                 # SHAP interpretability analysis
│   ├── inference.py                  # BDFSPredictor singleton for live inference
│   ├── eda_visualizer.py             # Exploratory data analysis plots
│   └── models.py                     # Helper model definitions
│
├── backend/                          # FastAPI REST API
│   ├── main.py                       # API server & route handlers
│   ├── schemas.py                    # Pydantic request/response models
│   └── requirements.txt              # Backend dependencies
│
├── frontend/                         # React + Vite + TypeScript UI
│   ├── src/
│   │   ├── App.tsx                   # Main application component
│   │   ├── api.ts                    # API client
│   │   ├── main.tsx                  # Entry point
│   │   └── styles.css                # Styling
│   ├── package.json
│   └── vite.config.ts
│
├── models/                           # Saved trained models
│   ├── xgb_best.pkl                  # XGBoost (537 KB) ⭐
│   ├── rf_best.pkl                   # Random Forest (99.3 MB)
│   ├── lr_best.pkl                   # Logistic Regression (1.1 KB)
│   ├── svm_best.pkl                  # SVM (1.3 MB)
│   └── knn_best.pkl                  # KNN (30.1 MB)
│
├── data/
│   ├── raw/                          # bdfs_raw.csv (150K records)
│   ├── processed/                    # Train / Val / Test splits
│   └── splits/                       # Scaler, feature names, split indices
│
├── figures/                          # Generated visualizations (20 PNGs)
├── results/                          # Evaluation metrics CSVs
├── notebooks/                        # Jupyter notebooks (EDA)
├── paper/                            # Research paper drafts
│
├── app.py                            # Streamlit dashboard
├── config.py                         # Centralized configuration & constants
├── run_all.py                        # Full pipeline orchestrator
├── create_stunning_curves.py         # Publication-quality ROC/PR curves
└── README.md
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for understanding human cognition**

*Developed as part of the Microsoft AI Project*

</div>
