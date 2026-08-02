"""
BDFS — Adım 6: Model Değerlendirmesi (Evaluation)
==================================================
Eğitilen 5 modelin test seti üzerinde değerlendirilmesi,
metrik hesaplamaları, Confusion Matrix, ROC ve PR eğrileri
çizimi ve McNemar istatistiksel testi.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from statsmodels.stats.contingency_tables import mcnemar

# Proje kökünü ayarla
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)
import config

os.makedirs(config.FIGURES_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("BDFS — ADIM 6: MODEL DEĞERLENDİRMESİ")
print("=" * 60)

# Veri yükle
print("Test verisi yükleniyor...")
data = joblib.load(config.SPLIT_INDICES_PATH)
X_test = data['X_test']
y_test = data['y_test']

# NaN/Inf temizliği
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Test Seti Boyutu: {X_test.shape}, Pozitif Sınıf Oranı: {y_test.mean():.4f}")

# Modelleri yükle
model_names = ['LR', 'RF', 'XGB', 'SVM', 'KNN']
models = {}
for name in model_names:
    file_path = f"{config.MODELS_DIR}{name.lower()}_best.pkl"
    if os.path.exists(file_path):
        models[name] = joblib.load(file_path)
    else:
        print(f"UYARI: {name} modeli bulunamadı ({file_path})")

results = []
predictions = {}

# Grafikler için figürler hazırla (Stunning Theme)
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f8f9fa", "grid.color": "#e9ecef"})
plt.figure(figsize=(11, 9))
plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=16, fontweight='bold', pad=20)
plt.plot([0, 1], [0, 1], color='#adb5bd', linestyle='--', lw=2, label='Random Guess')

fig_pr, ax_pr = plt.subplots(figsize=(11, 9))
ax_pr.set_title("Precision-Recall (PR) Curve Comparison", fontsize=16, fontweight='bold', pad=20)

colors = {'LR': '#4361ee', 'RF': '#2a9d8f', 'XGB': '#e63946', 'SVM': '#7209b7', 'KNN': '#fca311'}

# Modelleri değerlendir
for name, model in models.items():
    print(f"\n[{name}] modeli değerlendiriliyor...")
    
    # Tahminler
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Tahminleri McNemar testi için sakla
    predictions[name] = y_pred
    
    # Metrikler
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc,
        'PR-AUC': pr_auc
    })
    
    print(f"  ROC-AUC: {auc:.4f} | F1: {f1:.4f} | PR-AUC: {pr_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{config.FIGURES_DIR}{name.lower()}_confusion_matrix.png", dpi=300)
    plt.close()
    
    # ROC Curve verisi
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(1) # ROC figürüne dön
    plt.plot(fpr, tpr, color=colors.get(name, 'black'), lw=2, label=f'{name} (AUC = {auc:.3f})')
    
    # PR Curve verisi
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    ax_pr.plot(recall_curve, precision_curve, color=colors.get(name, 'black'), lw=2, label=f'{name} (PR-AUC = {pr_auc:.3f})')

# ROC Figürünü Kaydet
plt.figure(1)
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.legend(loc='lower right', frameon=True, fontsize=11, shadow=True, fancybox=True)
plt.tight_layout()
plt.savefig(f"{config.FIGURES_DIR}roc_curves_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# PR Figürünü Kaydet
ax_pr.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax_pr.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax_pr.legend(loc='upper right', frameon=True, fontsize=11, shadow=True, fancybox=True)
fig_pr.tight_layout()
fig_pr.savefig(f"{config.FIGURES_DIR}pr_curves_comparison.png", dpi=300, bbox_inches='tight')
plt.close(fig_pr)

# Sonuçları Kaydet
results_df = pd.DataFrame(results).round(4)
results_df.to_csv(f"{config.RESULTS_DIR}model_evaluation_metrics.csv", index=False)

print("\n" + "=" * 60)
print("TEST SETİ PERFORMANS ÖZETİ")
print("=" * 60)
print(results_df.to_string(index=False))

# McNemar Testi (XGB vs LR)
if 'XGB' in predictions and 'LR' in predictions:
    print("\n" + "=" * 60)
    print("MCNEMAR TESTİ (XGBoost vs Logistic Regression)")
    print("=" * 60)
    
    y_pred_xgb = predictions['XGB']
    y_pred_lr = predictions['LR']
    
    # Contingency Table oluştur
    # b: XGB doğru, LR yanlış
    # c: XGB yanlış, LR doğru
    b = np.sum((y_pred_xgb == y_test) & (y_pred_lr != y_test))
    c = np.sum((y_pred_xgb != y_test) & (y_pred_lr == y_test))
    a = np.sum((y_pred_xgb == y_test) & (y_pred_lr == y_test))
    d = np.sum((y_pred_xgb != y_test) & (y_pred_lr != y_test))
    
    contingency_table = [[a, b], [c, d]]
    print("Contingency Table:")
    print(f"XGB Doğru  | LR Doğru: {a} | LR Yanlış: {b}")
    print(f"XGB Yanlış | LR Doğru: {c} | LR Yanlış: {d}")
    
    # Testi uygula
    result = mcnemar(contingency_table, exact=False, correction=True)
    
    print(f"\nMcNemar Test Statistic: {result.statistic:.4f}")
    print(f"P-value: {result.pvalue:.4e}")
    
    if result.pvalue < 0.05:
        print("Sonuç: XGBoost ile Logistic Regression arasında İSTATİSTİKSEL OLARAK ANLAMLI bir fark vardır (p < 0.05).")
    else:
        print("Sonuç: İki model arasında istatistiksel olarak anlamlı bir fark YOKTUR (p >= 0.05).")

print("\n✓ Adım 6 tamamlandı — Değerlendirme raporları ve grafikler kaydedildi.")
print(f"  Metrikler: {config.RESULTS_DIR}model_evaluation_metrics.csv")
print(f"  Grafikler: {config.FIGURES_DIR}")
