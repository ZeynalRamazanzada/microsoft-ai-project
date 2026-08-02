"""
BDFS — EDA ve Görselleştirme Modülü
=====================================
Ham veriyi yükler, dağılımları analiz eder, görselleri üretir.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Görsel ayarları
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.facecolor': 'white'
})
sns.set_style("whitegrid")


def load_data():
    """Ham veriyi yükle ve temel bilgileri yazdır."""
    df = pd.read_csv(config.RAW_DATA_PATH)
    print(f"Veri yüklendi: {df.shape[0]} satır × {df.shape[1]} sütun")
    return df


def print_summary_stats(df):
    """Pandas describe() ile özet istatistikler."""
    print("\n" + "=" * 60)
    print("ÖZET İSTATİSTİKLER (describe)")
    print("=" * 60)
    print(df.describe().round(4).to_string())


def missing_outlier_report(df):
    """Her feature için eksik değer ve outlier yüzdesi raporu."""
    print("\n" + "=" * 60)
    print("EKSİK DEĞER VE OUTLIER RAPORU")
    print("=" * 60)
    
    excluded = ['participant_id', 'trial', 'fatigue_class', 'session_load']
    numeric_cols = [c for c in df.columns if c not in excluded]
    
    report_rows = []
    for col in numeric_cols:
        n_total = len(df)
        # Eksik değer
        n_missing = df[col].isnull().sum()
        pct_missing = 100.0 * n_missing / n_total
        
        # Outlier (IQR yöntemi)
        valid = df[col].dropna()
        Q1 = valid.quantile(0.25)
        Q3 = valid.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outlier = ((valid < lower) | (valid > upper)).sum()
        pct_outlier = 100.0 * n_outlier / len(valid)
        
        report_rows.append({
            'Feature': col,
            'Missing': n_missing,
            'Missing%': round(pct_missing, 2),
            'Outlier': n_outlier,
            'Outlier%': round(pct_outlier, 2),
            'Mean': round(valid.mean(), 4),
            'Std': round(valid.std(), 4)
        })
    
    report_df = pd.DataFrame(report_rows)
    print(report_df.to_string(index=False))
    report_df.to_csv("results/missing_outlier_report.csv", index=False)
    print("\n→ Rapor kaydedildi: results/missing_outlier_report.csv")
    return report_df


def plot_feature_distributions(df):
    """Her feature'ın dağılımını çiz (histogram, 0/1 sınıf ayrı renkle). 4×5 grid."""
    excluded = ['participant_id', 'trial', 'session_load', 'fatigue_class']
    features = [c for c in df.columns if c not in excluded]
    
    n_features = len(features)
    n_cols = 5
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, n_rows * 4))
    axes = axes.flatten()
    
    colors = {0: '#3498db', 1: '#e74c3c'}
    labels = {0: 'Non-fatigued (0)', 1: 'Fatigued (1)'}
    
    for i, feat in enumerate(features):
        ax = axes[i]
        for cls in [0, 1]:
            data = df[df['fatigue_class'] == cls][feat].dropna()
            ax.hist(data, bins=50, alpha=0.6, color=colors[cls],
                    label=labels[cls], density=True, edgecolor='none')
        ax.set_title(feat, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Density')
        if i == 0:
            ax.legend(fontsize=8)
    
    # Boş subplot'ları gizle
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('BDFS — Feature Dağılımları (Sınıf Bazlı)', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{config.FIGURES_DIR}01_feature_distributions.png", bbox_inches='tight')
    plt.close()
    print("✓ Görsel 1: Feature dağılımları kaydedildi")


def plot_correlation_matrix(df):
    """Korelasyon matrisi — seaborn heatmap."""
    excluded = ['participant_id', 'trial', 'fatigue_class']
    numeric_cols = [c for c in df.columns if c not in excluded and df[c].dtype != 'object']
    
    corr = df[numeric_cols].corr(method='pearson')
    
    fig, ax = plt.subplots(figsize=(18, 15))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True, fmt='.2f',
                square=True, linewidths=0.5, ax=ax,
                annot_kws={'size': 7}, vmin=-1, vmax=1,
                cbar_kws={'label': 'Pearson Correlation'})
    
    ax.set_title('BDFS — Pearson Korelasyon Matrisi', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{config.FIGURES_DIR}02_correlation_matrix.png", bbox_inches='tight')
    plt.close()
    print("✓ Görsel 2: Korelasyon matrisi kaydedildi")
    
    # Yüksek korelasyon çiftlerini raporla
    print("\n  Yüksek korelasyon çiftleri (|r| > 0.5):")
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                print(f"    {corr.columns[i]} ↔ {corr.columns[j]}: r = {r:.3f}")


def plot_class_distribution(df):
    """Class distribution bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    counts = df['fatigue_class'].value_counts().sort_index()
    colors = ['#3498db', '#e74c3c']
    labels_map = ['Non-fatigued (0)', 'Fatigued (1)']
    
    bars = axes[0].bar(labels_map, counts.values, color=colors, edgecolor='white', linewidth=1.5)
    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 500,
                     f'{count:,}\n({100 * count / len(df):.1f}%)',
                     ha='center', va='bottom', fontweight='bold', fontsize=12)
    axes[0].set_title('Sınıf Dağılımı', fontweight='bold', fontsize=14)
    axes[0].set_ylabel('Kayıt Sayısı')
    axes[0].set_ylim(0, max(counts.values) * 1.15)
    
    # Pie chart
    axes[1].pie(counts.values, labels=labels_map, colors=colors,
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12},
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Sınıf Oranları', fontweight='bold', fontsize=14)
    
    plt.suptitle('BDFS — Class Imbalance Analizi', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.FIGURES_DIR}03_class_distribution.png", bbox_inches='tight')
    plt.close()
    print("✓ Görsel 3: Class distribution kaydedildi")


def plot_fatigue_trajectory(df):
    """Tek bir katılımcının 50 tur boyunca feature + label değişimi."""
    # Katılımcı 42'yi seç (deterministik)
    p_data = df[df['participant_id'] == 42].copy()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # RT Mean
    axes[0].plot(p_data['trial'], p_data['rt_mean'], 'o-', color='#2980b9', markersize=4, linewidth=1.5)
    axes[0].set_ylabel('RT Mean (s)')
    axes[0].set_title('Katılımcı #42 — Fatigue Trajectory', fontweight='bold', fontsize=14)
    
    # Pref Reversal Rate
    axes[1].plot(p_data['trial'], p_data['pref_reversal_rate'], 's-', color='#e67e22', markersize=4, linewidth=1.5)
    axes[1].set_ylabel('Pref. Reversal Rate')
    
    # Label (fatigue_class)
    colors_label = ['#27ae60' if x == 0 else '#c0392b' for x in p_data['fatigue_class']]
    axes[2].bar(p_data['trial'], p_data['fatigue_class'], color=colors_label, alpha=0.8)
    axes[2].set_ylabel('Fatigue Label')
    axes[2].set_xlabel('Tur Numarası')
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(['Non-fatigued', 'Fatigued'])
    
    # Fatigue bölgelerini vurgula
    for ax in axes[:2]:
        for _, row in p_data.iterrows():
            if row['fatigue_class'] == 1:
                ax.axvspan(row['trial'] - 0.5, row['trial'] + 0.5, alpha=0.1, color='red')
    
    plt.tight_layout()
    plt.savefig(f"{config.FIGURES_DIR}12_fatigue_trajectory.png", bbox_inches='tight')
    plt.close()
    print("✓ Görsel 12: Fatigue trajectory kaydedildi")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("BDFS — ADIM 2: EDA VE GÖRSELLEŞTİRME")
    print("=" * 60)
    
    # Veri yükle
    df = load_data()
    
    # Özet istatistikler
    print_summary_stats(df)
    
    # Eksik değer ve outlier raporu
    missing_outlier_report(df)
    
    # Görseller
    print("\n" + "=" * 60)
    print("GÖRSELLER ÜRETİLİYOR")
    print("=" * 60)
    
    plot_feature_distributions(df)
    plot_correlation_matrix(df)
    plot_class_distribution(df)
    plot_fatigue_trajectory(df)
    
    print("\n" + "=" * 60)
    print("✓ Adım 2 tamamlandı — EDA ve görselleştirme başarılı!")
    print("=" * 60)
