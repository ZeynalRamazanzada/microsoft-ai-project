"""
BDFS — Adım 7: Ablation Study
==============================
Farklı feature setlerinin XGBoost performansı üzerindeki etkisinin test edilmesi.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

import warnings
warnings.filterwarnings('ignore')

# Proje kökünü ayarla
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)
import config

os.makedirs(config.FIGURES_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("BDFS — ADIM 7: ABLATION STUDY")
print("=" * 60)

# Veri yükle
data = joblib.load(config.SPLIT_INDICES_PATH)
X_train = np.nan_to_num(data['X_train'], nan=0.0, posinf=0.0, neginf=0.0)
y_train = data['y_train']
X_test = np.nan_to_num(data['X_test'], nan=0.0, posinf=0.0, neginf=0.0)
y_test = data['y_test']
features = np.array(data['feature_names'])

print(f"Toplam özellik (Full): {len(features)}")

# En iyi model parametrelerini yükle
xgb_model = joblib.load(f"{config.MODELS_DIR}xgb_best.pkl")
best_params = xgb_model.get_params()
print(f"Referans XGBoost Parametreleri kullanılıyor.")

# ---------------------------------------------------------
# Feature Setlerini Tanımla
# ---------------------------------------------------------
configs = {}

# Config A: Full
configs['Full Model'] = features.tolist()

# Config B: No DDM
ddm_feats = ['ez_drift_rate', 'ez_boundary', 'ez_nondecision', 'drift_boundary_ratio', 'ddm_fatigue_index']
configs['No DDM'] = [f for f in features if f not in ddm_feats]

# Config C: No Temporal
temporal_feats = ['fatigue_slope', 'session_position', 'accuracy_decay_rate', 'rt_slope', 
                  'rolling_incon_5', 'rolling_incon_10', 'incon_acceleration', 'temporal_load_interaction']
configs['No Temporal'] = [f for f in features if f not in temporal_feats]

# Config D: Baseline
baseline_feats = ['rt_mean', 'rt_std', 'pref_reversal_rate', 'rolling_incon_5', 'choice_entropy']
# Baseline'da sadece veri setinde var olanları al
configs['Baseline'] = [f for f in features if f in baseline_feats]

# ---------------------------------------------------------
# Modelleri Eğit ve Değerlendir
# ---------------------------------------------------------
results = []
models = {}

for config_name, config_features in configs.items():
    print(f"\nDeğerlendiriliyor: {config_name} ({len(config_features)} özellik)")
    
    # Feature indekslerini bul
    idx = [np.where(features == f)[0][0] for f in config_features]
    
    # Alt kümeyi oluştur
    X_train_sub = X_train[:, idx]
    X_test_sub = X_test[:, idx]
    
    # Modeli eğit
    model = XGBClassifier(**best_params)
    model.fit(X_train_sub, y_train)
    models[config_name] = model
    
    # Tahmin
    y_pred = model.predict(X_test_sub)
    y_prob = model.predict_proba(X_test_sub)[:, 1]
    
    # Metrikler
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        'Config': config_name,
        'Num_Features': len(config_features),
        'ROC_AUC': auc,
        'F1_Score': f1
    })
    print(f"  ROC-AUC: {auc:.4f} | F1: {f1:.4f}")

# ---------------------------------------------------------
# Sonuçları Formatla ve Kaydet
# ---------------------------------------------------------
df_results = pd.DataFrame(results)
full_auc = df_results[df_results['Config'] == 'Full Model']['ROC_AUC'].values[0]
df_results['Delta_AUC'] = df_results['ROC_AUC'] - full_auc

print("\n" + "=" * 60)
print("ABLATION STUDY SONUÇLARI")
print("=" * 60)
print(df_results.round(4).to_string(index=False))

df_results.round(4).to_csv(f"{config.RESULTS_DIR}ablation_results.csv", index=False)

# ---------------------------------------------------------
# Görselleştirme (Bar Chart)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'] # Green, Blue, Orange, Red
bars = plt.bar(df_results['Config'], df_results['ROC_AUC'], color=colors, alpha=0.8)

# Değerleri bar üstüne yaz
for bar, auc, delta in zip(bars, df_results['ROC_AUC'], df_results['Delta_AUC']):
    height = bar.get_height()
    
    # AUC text
    plt.text(bar.get_x() + bar.get_width()/2., height - 0.05,
             f'{auc:.3f}', ha='center', va='bottom', color='white', fontweight='bold', fontsize=12)
    
    # Delta text
    if delta < 0:
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{delta*100:.1f}%', ha='center', va='bottom', color='red', fontweight='bold')

plt.title('Ablation Study: XGBoost ROC-AUC Performance', fontsize=14, pad=15)
plt.ylabel('ROC-AUC Score')
plt.ylim(0.5, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig(f"{config.FIGURES_DIR}11_ablation_study.png", dpi=300)
plt.close()

print("\n✓ Adım 7 tamamlandı — Ablation study sonuçları ve grafik kaydedildi.")
