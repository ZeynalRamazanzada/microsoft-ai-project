"""
BDFS — Model Eğitim Modülü
==============================
5 ML modeli: LR, RF, XGBoost, SVM, KNN
Her biri sklearn Pipeline + GridSearch/RandomizedSearch ile tune edilir.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_data():
    """Feature engineering sonrası verileri yükle."""
    data = joblib.load(config.SPLIT_INDICES_PATH)
    print(f"Train: {data['X_train'].shape}, Val: {data['X_val'].shape}, Test: {data['X_test'].shape}")
    return data


def train_logistic_regression(X_train, y_train, cv):
    """Model 1: Logistic Regression — GridSearchCV."""
    print("\n" + "=" * 60)
    print("MODEL 1: LOGISTIC REGRESSION")
    print("=" * 60)

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': ['balanced', None],
        'max_iter': [2000]
    }

    lr = LogisticRegression(random_state=config.RANDOM_SEED)
    grid = GridSearchCV(lr, param_grid, cv=cv, scoring='roc_auc',
                        n_jobs=-1, verbose=0)

    start = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"\n  En iyi parametreler: {grid.best_params_}")
    print(f"  En iyi ROC-AUC (CV): {grid.best_score_:.4f}")
    print(f"  Süre: {elapsed:.1f}s")
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def train_random_forest(X_train, y_train, cv):
    """Model 2: Random Forest — RandomizedSearchCV (n_iter=50)."""
    print("\n" + "=" * 60)
    print("MODEL 2: RANDOM FOREST")
    print("=" * 60)

    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.5],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }

    rf = RandomForestClassifier(random_state=config.RANDOM_SEED, n_jobs=-1)
    search = RandomizedSearchCV(rf, param_dist, n_iter=50, cv=cv,
                                scoring='roc_auc', n_jobs=1, verbose=1,
                                random_state=config.RANDOM_SEED)

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"\n  En iyi parametreler: {search.best_params_}")
    print(f"  En iyi ROC-AUC (CV): {search.best_score_:.4f}")
    print(f"  Süre: {elapsed:.1f}s")
    return search.best_estimator_, search.best_params_, search.best_score_


def train_xgboost(X_train, y_train, cv):
    """Model 3: XGBoost — RandomizedSearchCV (n_iter=60)."""
    print("\n" + "=" * 60)
    print("MODEL 3: XGBOOST")
    print("=" * 60)

    # Imbalance oranı hesapla
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_ratio = round(neg_count / pos_count, 2)

    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [1, 5, 10],
        'scale_pos_weight': [1, 2, scale_ratio]
    }

    xgb = XGBClassifier(random_state=config.RANDOM_SEED, eval_metric='logloss',
                         use_label_encoder=False, verbosity=0, n_jobs=-1)
    search = RandomizedSearchCV(xgb, param_dist, n_iter=60, cv=cv,
                                scoring='roc_auc', n_jobs=1, verbose=1,
                                random_state=config.RANDOM_SEED)

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"\n  En iyi parametreler: {search.best_params_}")
    print(f"  En iyi ROC-AUC (CV): {search.best_score_:.4f}")
    print(f"  Süre: {elapsed:.1f}s")
    return search.best_estimator_, search.best_params_, search.best_score_


def train_svm(X_train, y_train, cv):
    """Model 4: SVM (RBF) — GridSearchCV + 30K subsample."""
    print("\n" + "=" * 60)
    print("MODEL 4: SVM (RBF)")
    print("=" * 60)

    # 30.000 subsample (SVM O(n²) karmaşıklık)
    n_sub = 30000
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = rng.choice(len(X_train), size=min(n_sub, len(X_train)), replace=False)
    X_sub = X_train[idx]
    y_sub = y_train[idx]
    print(f"  Subsample: {len(X_train)} → {len(X_sub)}")

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.001],
        'class_weight': ['balanced', None]
    }

    svm = SVC(kernel='rbf', probability=True, random_state=config.RANDOM_SEED)
    grid = GridSearchCV(svm, param_grid, cv=cv, scoring='roc_auc',
                        n_jobs=-1, verbose=1)

    start = time.time()
    grid.fit(X_sub, y_sub)
    elapsed = time.time() - start

    print(f"\n  En iyi parametreler: {grid.best_params_}")
    print(f"  En iyi ROC-AUC (CV): {grid.best_score_:.4f}")
    print(f"  Süre: {elapsed:.1f}s")
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def train_knn(X_train, y_train, cv):
    """Model 5: KNN — GridSearchCV."""
    print("\n" + "=" * 60)
    print("MODEL 5: K-NEAREST NEIGHBORS")
    print("=" * 60)

    param_grid = {
        'n_neighbors': [3, 5, 7, 11, 15, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }

    knn = KNeighborsClassifier(n_jobs=-1)
    grid = GridSearchCV(knn, param_grid, cv=cv, scoring='roc_auc',
                        n_jobs=1, verbose=1)

    start = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"\n  En iyi parametreler: {grid.best_params_}")
    print(f"  En iyi ROC-AUC (CV): {grid.best_score_:.4f}")
    print(f"  Süre: {elapsed:.1f}s")
    return grid.best_estimator_, grid.best_params_, grid.best_score_


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("BDFS — ADIM 5: MODEL EĞİTİMİ")
    print("=" * 60)

    data = load_data()
    X_train = data['X_train']
    y_train = data['y_train']
    feature_names = data['feature_names']

    # NaN/Inf temizliği (güvenlik)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    # Hyperparameter tuning için 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

    results = {}
    total_start = time.time()

    # Model 1: Logistic Regression
    lr_model, lr_params, lr_score = train_logistic_regression(X_train, y_train, cv)
    results['LR'] = {'model': lr_model, 'params': lr_params, 'cv_auc': lr_score}
    joblib.dump(lr_model, f"{config.MODELS_DIR}lr_best.pkl")
    print(f"  → Kaydedildi: {config.MODELS_DIR}lr_best.pkl")

    # Model 2: Random Forest
    rf_model, rf_params, rf_score = train_random_forest(X_train, y_train, cv)
    results['RF'] = {'model': rf_model, 'params': rf_params, 'cv_auc': rf_score}
    joblib.dump(rf_model, f"{config.MODELS_DIR}rf_best.pkl")
    print(f"  → Kaydedildi: {config.MODELS_DIR}rf_best.pkl")

    # Model 3: XGBoost
    xgb_model, xgb_params, xgb_score = train_xgboost(X_train, y_train, cv)
    results['XGB'] = {'model': xgb_model, 'params': xgb_params, 'cv_auc': xgb_score}
    joblib.dump(xgb_model, f"{config.MODELS_DIR}xgb_best.pkl")
    print(f"  → Kaydedildi: {config.MODELS_DIR}xgb_best.pkl")

    # Model 4: SVM
    svm_model, svm_params, svm_score = train_svm(X_train, y_train, cv)
    results['SVM'] = {'model': svm_model, 'params': svm_params, 'cv_auc': svm_score}
    joblib.dump(svm_model, f"{config.MODELS_DIR}svm_best.pkl")
    print(f"  → Kaydedildi: {config.MODELS_DIR}svm_best.pkl")

    # Model 5: KNN
    knn_model, knn_params, knn_score = train_knn(X_train, y_train, cv)
    results['KNN'] = {'model': knn_model, 'params': knn_params, 'cv_auc': knn_score}
    joblib.dump(knn_model, f"{config.MODELS_DIR}knn_best.pkl")
    print(f"  → Kaydedildi: {config.MODELS_DIR}knn_best.pkl")

    total_elapsed = time.time() - total_start

    # Özet tablo
    print("\n" + "=" * 60)
    print("MODEL KARŞILAŞTIRMA (CV ROC-AUC)")
    print("=" * 60)
    summary = pd.DataFrame([
        {'Model': name, 'CV_ROC_AUC': f"{info['cv_auc']:.4f}", 'Best_Params': str(info['params'])}
        for name, info in results.items()
    ])
    print(summary.to_string(index=False))
    summary.to_csv(f"{config.RESULTS_DIR}model_hyperparams.csv", index=False)

    print(f"\nToplam süre: {total_elapsed/60:.1f} dakika")
    print("\n" + "=" * 60)
    print("✓ Adım 5 tamamlandı — Model eğitimi başarılı!")
    print("=" * 60)
