"""
BDFS — Adım 5: Model Eğitimi (Hız İçin Çok Daha Optimize Edilmiş Versiyon)
=============================================================================
5 ML modeli sırasıyla eğitilir, her biri bitince kaydedilir.
Hızlandırmak için n_iter ve search space küçültülmüştür.
"""

import os
import sys
import warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import joblib
import time

# Proje kökünü ayarla
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)
import config

os.makedirs(config.MODELS_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("BDFS — ADIM 5: MODEL EĞİTİMİ (HIZLANDIRILMIŞ)")
print("=" * 60)

# Veri yükle
data = joblib.load(config.SPLIT_INDICES_PATH)
X_train = data['X_train']
y_train = data['y_train']
feature_names = data['feature_names']

# NaN/Inf temizliği
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Train: {X_train.shape}, Features: {len(feature_names)}")
print(f"Sınıf dağılımı: 0={int((y_train==0).sum())}, 1={int((y_train==1).sum())}")

# Tuning için 3-fold CV
cv_tune = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_SEED)

results = {}
total_start = time.time()

# ============================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================
print("\n" + "=" * 60)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 60)

param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
    'class_weight': ['balanced', None],
    'max_iter': [1000]
}
lr = LogisticRegression(random_state=config.RANDOM_SEED)
grid = GridSearchCV(lr, param_grid, cv=cv_tune, scoring='roc_auc', n_jobs=-1, verbose=0)
t0 = time.time()
grid.fit(X_train, y_train)
t1 = time.time()
print(f"  Best params: {grid.best_params_}")
print(f"  Best ROC-AUC (CV): {grid.best_score_:.4f}")
print(f"  Süre: {t1-t0:.1f}s")
joblib.dump(grid.best_estimator_, f"{config.MODELS_DIR}lr_best.pkl")
results['LR'] = {'params': grid.best_params_, 'cv_auc': grid.best_score_}
print("  ✓ LR kaydedildi")
sys.stdout.flush()

# ============================================================
# MODEL 2: RANDOM FOREST
# ============================================================
print("\n" + "=" * 60)
print("MODEL 2: RANDOM FOREST")
print("=" * 60)

param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', None]
}
rf = RandomForestClassifier(random_state=config.RANDOM_SEED, n_jobs=-1)
search = RandomizedSearchCV(rf, param_dist, n_iter=10, cv=cv_tune,
                            scoring='roc_auc', n_jobs=1, verbose=2,
                            random_state=config.RANDOM_SEED)
t0 = time.time()
search.fit(X_train, y_train)
t1 = time.time()
print(f"\n  Best params: {search.best_params_}")
print(f"  Best ROC-AUC (CV): {search.best_score_:.4f}")
print(f"  Süre: {(t1-t0)/60:.1f} dakika")
joblib.dump(search.best_estimator_, f"{config.MODELS_DIR}rf_best.pkl")
results['RF'] = {'params': search.best_params_, 'cv_auc': search.best_score_}
print("  ✓ RF kaydedildi")
sys.stdout.flush()

# ============================================================
# MODEL 3: XGBOOST
# ============================================================
print("\n" + "=" * 60)
print("MODEL 3: XGBOOST")
print("=" * 60)

neg_count = int((y_train == 0).sum())
pos_count = int((y_train == 1).sum())
scale_ratio = round(neg_count / pos_count, 2)
print(f"  scale_pos_weight hesaplandı: {scale_ratio}")

param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, scale_ratio]
}
xgb = XGBClassifier(random_state=config.RANDOM_SEED, eval_metric='logloss',
                     use_label_encoder=False, verbosity=0, n_jobs=-1,
                     tree_method='hist')
search = RandomizedSearchCV(xgb, param_dist, n_iter=15, cv=cv_tune,
                            scoring='roc_auc', n_jobs=1, verbose=2,
                            random_state=config.RANDOM_SEED)
t0 = time.time()
search.fit(X_train, y_train)
t1 = time.time()
print(f"\n  Best params: {search.best_params_}")
print(f"  Best ROC-AUC (CV): {search.best_score_:.4f}")
print(f"  Süre: {(t1-t0)/60:.1f} dakika")
joblib.dump(search.best_estimator_, f"{config.MODELS_DIR}xgb_best.pkl")
results['XGB'] = {'params': search.best_params_, 'cv_auc': search.best_score_}
print("  ✓ XGB kaydedildi")
sys.stdout.flush()

# ============================================================
# MODEL 4: SVM (RBF)
# ============================================================
print("\n" + "=" * 60)
print("MODEL 4: SVM (RBF)")
print("=" * 60)

# Daha da hızlandırmak için 10.000 subsample kullanıyoruz
n_sub = 10000
rng = np.random.default_rng(config.RANDOM_SEED)
idx = rng.choice(len(X_train), size=min(n_sub, len(X_train)), replace=False)
X_sub = X_train[idx]
y_sub = y_train[idx]
print(f"  Subsample: {len(X_train)} → {len(X_sub)}")

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'class_weight': ['balanced', None]
}
svm = SVC(kernel='rbf', probability=True, random_state=config.RANDOM_SEED)
grid = GridSearchCV(svm, param_grid, cv=cv_tune, scoring='roc_auc',
                    n_jobs=-1, verbose=2)
t0 = time.time()
grid.fit(X_sub, y_sub)
t1 = time.time()
print(f"\n  Best params: {grid.best_params_}")
print(f"  Best ROC-AUC (CV): {grid.best_score_:.4f}")
print(f"  Süre: {(t1-t0)/60:.1f} dakika")
joblib.dump(grid.best_estimator_, f"{config.MODELS_DIR}svm_best.pkl")
results['SVM'] = {'params': grid.best_params_, 'cv_auc': grid.best_score_}
print("  ✓ SVM kaydedildi")
sys.stdout.flush()

# ============================================================
# MODEL 5: K-NEAREST NEIGHBORS
# ============================================================
print("\n" + "=" * 60)
print("MODEL 5: K-NEAREST NEIGHBORS")
print("=" * 60)

param_grid = {
    'n_neighbors': [5, 11, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean'],
}
knn = KNeighborsClassifier(n_jobs=-1)
grid = GridSearchCV(knn, param_grid, cv=cv_tune, scoring='roc_auc',
                    n_jobs=1, verbose=2)
t0 = time.time()
grid.fit(X_train, y_train)
t1 = time.time()
print(f"\n  Best params: {grid.best_params_}")
print(f"  Best ROC-AUC (CV): {grid.best_score_:.4f}")
print(f"  Süre: {(t1-t0)/60:.1f} dakika")
joblib.dump(grid.best_estimator_, f"{config.MODELS_DIR}knn_best.pkl")
results['KNN'] = {'params': grid.best_params_, 'cv_auc': grid.best_score_}
print("  ✓ KNN kaydedildi")
sys.stdout.flush()

# ============================================================
# ÖZET TABLO
# ============================================================
total_elapsed = time.time() - total_start

print("\n" + "=" * 60)
print("MODEL KARŞILAŞTIRMA ÖZETİ (Hyperparameter Tuning CV)")
print("=" * 60)
summary_rows = []
for name, info in results.items():
    summary_rows.append({
        'Model': name,
        'CV_ROC_AUC': f"{info['cv_auc']:.4f}",
        'Best_Params': str(info['params'])
    })
summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))
summary.to_csv(f"{config.RESULTS_DIR}model_hyperparams.csv", index=False)

print(f"\nToplam süre: {total_elapsed/60:.1f} dakika")
print("\n" + "=" * 60)
print("✓ Adım 5 tamamlandı — 5 model başarıyla eğitildi!")
print("=" * 60)
