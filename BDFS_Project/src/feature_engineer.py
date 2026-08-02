"""
BDFS — Feature Engineering Modülü
====================================
Korelasyon analizi, VIF, Mutual Information, 6 yeni feature türetme ve RFE.
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config




def load_split_data():
    """Preprocessing sonrası kaydedilmiş verileri yükle."""
    data = joblib.load(config.SPLIT_INDICES_PATH)
    print(f"Veri yüklendi: Train={data['X_train'].shape}, Val={data['X_val'].shape}, Test={data['X_test'].shape}")
    print(f"Feature'lar ({len(data['feature_names'])}): {data['feature_names']}")
    return data


def correlation_analysis(X_train, feature_names):
    """Pearson korelasyon matrisi + yüksek korelasyon raporu."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("\n" + "-" * 50)
    print("ADIM 1: Pearson Korelasyon Analizi")
    print("-" * 50)

    df_train = pd.DataFrame(X_train, columns=feature_names)
    corr = df_train.corr(method='pearson')

    # Heatmap
    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(250, 10, as_cmap=True),
                center=0, annot=True, fmt='.2f', square=True, linewidths=0.5,
                ax=ax, annot_kws={'size': 7}, vmin=-1, vmax=1)
    ax.set_title('Feature Korelasyon Matrisi (Train Set — Scaled)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.FIGURES_DIR}02b_correlation_train.png", bbox_inches='tight')
    plt.close()

    # Yüksek korelasyon çiftleri
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                high_corr.append((corr.columns[i], corr.columns[j], round(r, 3)))

    print(f"\n  Yüksek korelasyon çiftleri (|r| > 0.5):")
    for f1, f2, r in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
        marker = "⚠" if abs(r) > 0.8 else " "
        print(f"  {marker} {f1:<30s} ↔ {f2:<30s} r = {r:+.3f}")

    n_critical = sum(1 for _, _, r in high_corr if abs(r) > 0.8)
    print(f"\n  |r| > 0.8 çift sayısı: {n_critical}")
    print("  ✓ Korelasyon analizi tamamlandı")
    return corr


def vif_analysis(X_train, feature_names):
    """VIF (Variance Inflation Factor) hesapla."""
    print("\n" + "-" * 50)
    print("ADIM 2: VIF Analizi")
    print("-" * 50)

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    df_train = pd.DataFrame(X_train, columns=feature_names)
    vif_data = []
    for i in range(len(feature_names)):
        try:
            vif_val = variance_inflation_factor(df_train.values, i)
        except Exception:
            vif_val = np.nan
        vif_data.append({'Feature': feature_names[i], 'VIF': round(vif_val, 2)})

    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    print(vif_df.to_string(index=False))

    problematic = vif_df[vif_df['VIF'] > 10]
    if len(problematic) > 0:
        print(f"\n  ⚠ VIF > 10 olan feature'lar ({len(problematic)}):")
        for _, row in problematic.iterrows():
            print(f"    {row['Feature']}: VIF = {row['VIF']}")
    else:
        print("\n  ✓ VIF > 10 olan feature yok — multicollinearity sorunsuz")

    return vif_df


def mutual_information_analysis(X_train, y_train, feature_names):
    """Mutual Information skorları hesapla + görselleştir."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.feature_selection import mutual_info_classif

    print("\n" + "-" * 50)
    print("ADIM 3: Mutual Information Analizi")
    print("-" * 50)

    mi_scores = mutual_info_classif(X_train, y_train, random_state=config.RANDOM_SEED, n_neighbors=5)
    mi_df = pd.DataFrame({'Feature': feature_names, 'MI_Score': mi_scores})
    mi_df = mi_df.sort_values('MI_Score', ascending=True)

    print("\n  MI Skorları (azalan sıra):")
    for _, row in mi_df.sort_values('MI_Score', ascending=False).iterrows():
        bar = "█" * int(row['MI_Score'] * 100)
        print(f"    {row['Feature']:<30s} {row['MI_Score']:.4f}  {bar}")

    # Görselleştirme
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(mi_df)))
    ax.barh(mi_df['Feature'], mi_df['MI_Score'], color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Mutual Information Score')
    ax.set_title('BDFS — Feature Mutual Information (Label ile)', fontsize=14, fontweight='bold')
    for i, (_, row) in enumerate(mi_df.iterrows()):
        ax.text(row['MI_Score'] + 0.001, i, f"{row['MI_Score']:.4f}", va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{config.FIGURES_DIR}04_mutual_information.png", bbox_inches='tight')
    plt.close()
    print("\n  ✓ Mutual information analizi tamamlandı")

    return mi_df.sort_values('MI_Score', ascending=False)


def derive_new_features(X, feature_names):
    """6 yeni feature türet (18'den 24'e + one-hot = 26'ya çıkar)."""
    df = pd.DataFrame(X, columns=feature_names)
    fn = {name: i for i, name in enumerate(feature_names)}

    # 1. rt_cv (Coefficient of Variation)
    rt_std_vals = df.iloc[:, fn['rt_std']] if 'rt_std' in fn else df['rt_std']
    rt_mean_vals = df.iloc[:, fn['rt_mean']] if 'rt_mean' in fn else df['rt_mean']
    df['rt_cv'] = rt_std_vals / (rt_mean_vals + 1e-6)

    # 2. incon_acceleration
    df['incon_acceleration'] = df['rolling_incon_5'] - df['rolling_incon_10']

    # 3. ddm_fatigue_index
    df['ddm_fatigue_index'] = (1.0 / (df['ez_drift_rate'] + 1e-6)) * df['ez_nondecision']

    # 4. temporal_load_interaction
    df['temporal_load_interaction'] = df['session_position'] * df['session_load']

    # 5. rt_kurtosis_normalized
    df['rt_kurtosis_normalized'] = (df['rt_kurtosis'] - 3) / (df['rt_std'] + 1e-6)

    # 6. decision_efficiency
    df['decision_efficiency'] = (1.0 - df['pref_reversal_rate']) / (df['rt_mean'] + 1e-6)

    new_feature_names = list(feature_names) + [
        'rt_cv', 'incon_acceleration', 'ddm_fatigue_index',
        'temporal_load_interaction', 'rt_kurtosis_normalized', 'decision_efficiency'
    ]
    return df.values, new_feature_names


def rfe_analysis(X_train, y_train, feature_names, n_select=15):
    """Recursive Feature Elimination (RF ile) — top-15 feature seç."""
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier

    print("\n" + "-" * 50)
    print(f"ADIM 5: RFE ile Top-{n_select} Feature Seçimi")
    print("-" * 50)

    rf = RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_SEED, n_jobs=-1)
    rfe = RFE(estimator=rf, n_features_to_select=n_select, step=1)
    rfe.fit(X_train, y_train)

    rfe_df = pd.DataFrame({
        'Feature': feature_names,
        'Selected': rfe.support_,
        'Ranking': rfe.ranking_
    }).sort_values('Ranking')

    print(f"\n  RFE Sıralaması:")
    for _, row in rfe_df.iterrows():
        marker = "★" if row['Selected'] else " "
        print(f"  {marker} {row['Feature']:<30s} Rank: {row['Ranking']}")

    selected = rfe_df[rfe_df['Selected']]['Feature'].tolist()
    print(f"\n  Seçilen {len(selected)} feature: {selected}")
    print("  ✓ RFE tamamlandı")
    return rfe_df, selected


def feature_selection_combined(mi_df, rfe_df, rf_importance_df, feature_names, n_top=15):
    """3 yöntemi birleştirerek 'core feature' setini belirle."""
    print("\n" + "-" * 50)
    print("ADIM 6: Kombine Feature Selection")
    print("-" * 50)

    mi_top = set(mi_df.head(n_top)['Feature'].tolist())
    rfe_top = set(rfe_df[rfe_df['Selected']]['Feature'].tolist())
    rf_top = set(rf_importance_df.head(n_top)['Feature'].tolist())

    scores = {}
    for f in feature_names:
        score = (1 if f in mi_top else 0) + (1 if f in rfe_top else 0) + (1 if f in rf_top else 0)
        scores[f] = score

    result = pd.DataFrame([{'Feature': f, 'Score': s, 'MI': f in mi_top,
                             'RFE': f in rfe_top, 'RF': f in rf_top}
                           for f, s in scores.items()]).sort_values('Score', ascending=False)

    core_features = result[result['Score'] >= 2]['Feature'].tolist()
    print(f"\n  Kombine sıralama (2+ yöntemde top-{n_top}):")
    for _, row in result.iterrows():
        marker = "★" if row['Score'] >= 2 else " "
        print(f"  {marker} {row['Feature']:<30s} Score={row['Score']} | MI={row['MI']} RFE={row['RFE']} RF={row['RF']}")

    print(f"\n  Core feature set ({len(core_features)}): {core_features}")
    return core_features, result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("BDFS — ADIM 4: FEATURE ENGINEERING")
    print("=" * 60)

    # Veri yükle
    data = load_split_data()
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    feature_names = data['feature_names']

    # 1. Korelasyon analizi
    corr = correlation_analysis(X_train, feature_names)

    # 2. VIF analizi
    try:
        vif_df = vif_analysis(X_train, feature_names)
    except ImportError:
        print("  ⚠ statsmodels yüklü değil, VIF atlandi")
        vif_df = None

    # 3. Mutual Information
    mi_df = mutual_information_analysis(X_train, y_train, feature_names)

    # 4. 6 yeni feature türet
    print("\n" + "-" * 50)
    print("ADIM 4: Yeni Feature Türetme")
    print("-" * 50)
    X_train_eng, new_features = derive_new_features(X_train, feature_names)
    X_val_eng, _ = derive_new_features(X_val, feature_names)
    X_test_eng, _ = derive_new_features(X_test, feature_names)
    print(f"  Feature sayısı: {len(feature_names)} → {len(new_features)}")
    print(f"  Yeni feature'lar: {new_features[len(feature_names):]}")
    print("  ✓ 6 yeni feature türetildi")

    # NaN/Inf temizliği
    for arr in [X_train_eng, X_val_eng, X_test_eng]:
        arr[np.isnan(arr)] = 0
        arr[np.isinf(arr)] = 0

    # 5. RFE (yeni feature'lar dahil)
    rfe_df, rfe_selected = rfe_analysis(X_train_eng, y_train, new_features, n_select=15)

    # RF Feature Importance
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif
    print("\n" + "-" * 50)
    print("RF Feature Importance")
    print("-" * 50)
    rf_model = RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_SEED, n_jobs=-1)
    rf_model.fit(X_train_eng, y_train)
    rf_imp_df = pd.DataFrame({
        'Feature': new_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    for _, row in rf_imp_df.iterrows():
        print(f"    {row['Feature']:<30s} {row['importance']:.4f}")

    # MI on new features
    mi_new = mutual_info_classif(X_train_eng, y_train, random_state=config.RANDOM_SEED)
    mi_new_df = pd.DataFrame({'Feature': new_features, 'MI_Score': mi_new}).sort_values('MI_Score', ascending=False)

    # 6. Kombine seçim
    core_features, selection_df = feature_selection_combined(mi_new_df, rfe_df, rf_imp_df, new_features)

    # Güncellenmiş verileri kaydet
    updated_data = {
        'X_train': X_train_eng, 'X_val': X_val_eng, 'X_test': X_test_eng,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'feature_names': new_features, 'core_features': core_features,
        'scaler': data['scaler'], 'selection_results': selection_df
    }
    joblib.dump(updated_data, config.SPLIT_INDICES_PATH)
    print(f"\n  → Güncellenmiş veri kaydedildi: {config.SPLIT_INDICES_PATH}")

    # Sonuçları CSV kaydet
    selection_df.to_csv(f"{config.RESULTS_DIR}feature_selection.csv", index=False)

    print("\n" + "=" * 60)
    print("✓ Adım 4 tamamlandı — Feature Engineering başarılı!")
    print("=" * 60)
