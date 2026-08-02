"""
BDFS — Adım 8: SHAP Yorumlanabilirliği (Interpretability)
=========================================================
En iyi modelin (XGBoost) tahminlerini yorumlamak için SHAP analizi.
Summary Plot, Dependence Plot ve Waterfall Plot üretilir.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

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
print("BDFS — ADIM 8: SHAP YORUMLANABİLİRLİĞİ")
print("=" * 60)

# Veri yükle
print("Veri ve model yükleniyor...")
data = joblib.load(config.SPLIT_INDICES_PATH)
X_test = np.nan_to_num(data['X_test'], nan=0.0, posinf=0.0, neginf=0.0)
y_test = data['y_test']
features = data['feature_names']

# Modeli yükle
xgb_model = joblib.load(f"{config.MODELS_DIR}xgb_best.pkl")

# SHAP hesaplaması çok uzun sürebileceğinden Test setinden 5.000 örneklik rastgele alt küme al
rng = np.random.default_rng(config.RANDOM_SEED)
sample_idx = rng.choice(X_test.shape[0], size=min(5000, X_test.shape[0]), replace=False)
X_shap = X_test[sample_idx]
y_shap = y_test[sample_idx]
print(f"SHAP Analizi için {X_shap.shape[0]} örneklem seçildi.")

# X_shap'ı DataFrame'e dönüştür (SHAP plotlarda feature isimleri görünmesi için)
X_shap_df = pd.DataFrame(X_shap, columns=features)

# TreeExplainer oluştur
print("SHAP TreeExplainer çalıştırılıyor (bu işlem birkaç dakika sürebilir)...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_shap_df)

# Feature önem sırasını çıkar (Mutlak SHAP değerlerinin ortalaması)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame({
    'Feature': features,
    'SHAP_Importance': mean_abs_shap
}).sort_values(by='SHAP_Importance', ascending=False)

top_feature = shap_importance.iloc[0]['Feature']
second_feature = shap_importance.iloc[1]['Feature']

print("\nEn Önemli 5 Feature (SHAP):")
print(shap_importance.head(5).to_string(index=False))
shap_importance.to_csv(f"{config.RESULTS_DIR}shap_feature_importance.csv", index=False)

# ---------------------------------------------------------
# 1. SHAP Summary Plot
# ---------------------------------------------------------
print("\nSHAP Summary Plot çiziliyor...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap_df, show=False)
plt.title("SHAP Summary Plot: Feature Impacts on Fatigue Prediction", pad=20, fontsize=14)
plt.tight_layout()
plt.savefig(f"{config.FIGURES_DIR}09_shap_summary.png", dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 2. SHAP Dependence Plot (En önemli 2 feature)
# ---------------------------------------------------------
print(f"SHAP Dependence Plot çiziliyor (Top feature: {top_feature})...")
plt.figure(figsize=(8, 6))
shap.dependence_plot(top_feature, shap_values, X_shap_df, show=False)
plt.title(f"SHAP Dependence Plot: {top_feature}", pad=20, fontsize=14)
plt.tight_layout()
plt.savefig(f"{config.FIGURES_DIR}10_shap_dependence_{top_feature}.png", dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 3. SHAP Waterfall Plot (Bireysel Tahmin Açıklaması)
# ---------------------------------------------------------
# Waterfall plot için explainer nesnesinden Explanation objesi elde etmeliyiz
# True Positive (Gerçekten yorgun ve model de yorgun demiş) olan yüksek olasılıklı bir örnek bulalım
probs = xgb_model.predict_proba(X_shap)[:, 1]
tp_indices = np.where((y_shap == 1) & (probs > 0.8))[0]

if len(tp_indices) > 0:
    target_idx = tp_indices[0]
    print(f"SHAP Waterfall Plot çiziliyor (Örnek index: {target_idx}, Olasılık: {probs[target_idx]:.3f})...")
    
    # SHAP nesnesini yeni formatta al
    explainer2 = shap.Explainer(xgb_model)
    shap_values2 = explainer2(X_shap_df.iloc[[target_idx]])
    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values2[0], max_display=10, show=False)
    plt.title(f"SHAP Waterfall Plot for High Risk Patient (Prob: {probs[target_idx]:.2f})", pad=20)
    plt.tight_layout()
    plt.savefig(f"{config.FIGURES_DIR}10b_shap_waterfall_tp.png", dpi=300, bbox_inches='tight')
    plt.close()

print("\n✓ Adım 8 tamamlandı — SHAP analizi yapıldı ve grafikler kaydedildi.")
