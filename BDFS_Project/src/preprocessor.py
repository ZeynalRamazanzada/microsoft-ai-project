"""
BDFS — Preprocessing Modülü
==============================
Ham veriyi temizler, ölçekler ve eğitim/doğrulama/test setlerine böler.

İşlem sırası:
  1. Median imputation (eksik değerler)
  2. IQR capping / Winsorization (aykırı değerler)
  3. One-hot encoding (task_complexity_level)
  4. Stratified train/val/test split (%70/%15/%15)
  5. StandardScaler (sadece train'e fit)
  6. SMOTE (sadece train setine)
  7. Kaydetme (CSV + pkl)
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_raw_data():
    """Ham veriyi yükle."""
    df = pd.read_csv(config.RAW_DATA_PATH)
    print(f"Ham veri yüklendi: {df.shape[0]} satır × {df.shape[1]} sütun")
    print(f"  Eksik değer toplam: {df.isnull().sum().sum()}")
    return df


def median_imputation(df):
    """
    Eksik değerleri median ile doldur.
    Kategorik feature'lar mode ile doldurulur.
    """
    print("\n" + "-" * 50)
    print("ADIM 1: Median Imputation")
    print("-" * 50)
    
    excluded = ['participant_id', 'trial', 'fatigue_class']
    numeric_cols = [c for c in df.columns if c not in excluded and df[c].dtype in ['float64', 'int64']]
    
    # Eksik değer öncesi rapor
    missing_before = df[numeric_cols].isnull().sum()
    print(f"  Eksik değer (önce): {missing_before.sum()}")
    
    # Median imputation
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Kontrol
    missing_after = df[numeric_cols].isnull().sum().sum()
    print(f"  Eksik değer (sonra): {missing_after}")
    assert missing_after == 0, "Imputation sonrası hâlâ eksik değer var!"
    print("  ✓ Tüm eksik değerler dolduruldu")
    
    return df


def iqr_capping(df):
    """
    IQR yöntemi ile aykırı değerleri kırp (Winsorization).
    Silmiyoruz, sınırlara çekiyoruz.
    """
    print("\n" + "-" * 50)
    print("ADIM 2: IQR Capping (Winsorization)")
    print("-" * 50)
    
    excluded = ['participant_id', 'trial', 'fatigue_class', 'session_load', 'task_complexity_level']
    numeric_cols = [c for c in df.columns if c not in excluded and df[c].dtype == 'float64']
    
    total_capped = 0
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # Kırpılacak değer sayısı
        n_below = (df[col] < lower).sum()
        n_above = (df[col] > upper).sum()
        n_capped = n_below + n_above
        
        if n_capped > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            total_capped += n_capped
    
    print(f"  Toplam kırpılan değer: {total_capped}")
    print("  ✓ Winsorization tamamlandı")
    
    return df


def one_hot_encoding(df):
    """
    task_complexity_level: {0, 1, 2} → One-Hot Encoding (drop='first').
    Dummy variable trap engellenir.
    """
    print("\n" + "-" * 50)
    print("ADIM 3: One-Hot Encoding")
    print("-" * 50)
    
    print(f"  task_complexity_level dağılımı (önce):")
    # Imputation sonrası float olabilir, en yakın int'e yuvarla
    df['task_complexity_level'] = df['task_complexity_level'].round().astype(int)
    # 0, 1, 2 aralığına kırp (outlier enjeksiyonu nedeniyle farklı değerler olabilir)
    df['task_complexity_level'] = df['task_complexity_level'].clip(0, 2)
    print(f"    {df['task_complexity_level'].value_counts().sort_index().to_dict()}")
    
    # One-hot encoding (drop_first=True → dummy trap engellenir)
    dummies = pd.get_dummies(df['task_complexity_level'], prefix='complexity', drop_first=True)
    dummies.columns = [f'complexity_{c}' for c in dummies.columns]
    
    # Orijinal sütunu kaldır, yenilerini ekle
    df = df.drop('task_complexity_level', axis=1)
    df = pd.concat([df, dummies.astype(int)], axis=1)
    
    print(f"  Yeni sütunlar: complexity_medium, complexity_high")
    print(f"  Toplam sütun sayısı: {df.shape[1]}")
    print("  ✓ One-hot encoding tamamlandı")
    
    return df


def stratified_split(df):
    """
    Stratified train/val/test split: %70/%15/%15.
    participant_id ve trial sütunları model feature'ı olarak kullanılmaz.
    """
    print("\n" + "-" * 50)
    print("ADIM 4: Stratified Train/Val/Test Split")
    print("-" * 50)
    
    # Feature ve label ayır
    drop_cols = ['participant_id', 'trial', 'fatigue_class']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].values
    y = df['fatigue_class'].values
    feature_names = feature_cols
    
    print(f"  Feature sayısı: {len(feature_names)}")
    print(f"  Feature'lar: {feature_names}")
    
    # ADIM 1: Test set ayır (%15)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        stratify=y, random_state=config.RANDOM_SEED
    )
    
    # ADIM 2: Validation set ayır (%15 orijinal ≈ %17.6 temp'ten)
    val_ratio = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio,
        stratify=y_temp, random_state=config.RANDOM_SEED
    )
    
    print(f"\n  Bölümleme sonuçları:")
    print(f"    Train : {X_train.shape[0]:>7,} kayıt ({100*X_train.shape[0]/len(X):.1f}%)")
    print(f"    Val   : {X_val.shape[0]:>7,} kayıt ({100*X_val.shape[0]/len(X):.1f}%)")
    print(f"    Test  : {X_test.shape[0]:>7,} kayıt ({100*X_test.shape[0]/len(X):.1f}%)")
    
    # Sınıf dağılımı kontrolü
    for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        pos = labels.sum()
        total = len(labels)
        print(f"    {name:5s} sınıf dağılımı: 0={total-pos} ({100*(total-pos)/total:.1f}%) | "
              f"1={pos} ({100*pos/total:.1f}%)")
    
    print("  ✓ Stratified split tamamlandı")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def apply_scaling(X_train, X_val, X_test):
    """
    StandardScaler — SADECE train set üzerinde fit edilir.
    Val ve test setleri transform edilir (fit değil!).
    """
    print("\n" + "-" * 50)
    print("ADIM 5: StandardScaler")
    print("-" * 50)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Train mean (örnek ilk 3): {X_train_scaled.mean(axis=0)[:3].round(4)}")
    print(f"  Train std  (örnek ilk 3): {X_train_scaled.std(axis=0)[:3].round(4)}")
    print("  ✓ Scaling tamamlandı (fit: sadece train)")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train):
    """
    SMOTE — Sadece train setine uygulanır.
    Validation ve test setlerine UYGULANMAZ (data leakage engeli).
    """
    print("\n" + "-" * 50)
    print("ADIM 6: SMOTE (Sadece Train Seti)")
    print("-" * 50)
    
    print(f"  SMOTE öncesi train sınıf dağılımı:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Sınıf {u}: {c:>7,} ({100*c/len(y_train):.1f}%)")
    
    smote = SMOTE(random_state=config.RANDOM_SEED)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    print(f"\n  SMOTE sonrası train sınıf dağılımı:")
    unique, counts = np.unique(y_train_bal, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Sınıf {u}: {c:>7,} ({100*c/len(y_train_bal):.1f}%)")
    
    print(f"\n  Train boyutu: {X_train.shape[0]:,} → {X_train_bal.shape[0]:,}")
    print("  ✓ SMOTE tamamlandı")
    
    return X_train_bal, y_train_bal


def save_all(X_train, X_val, X_test, y_train, y_val, y_test,
             X_train_bal, y_train_bal, feature_names, scaler):
    """Tüm verileri ve nesneleri kaydet."""
    print("\n" + "-" * 50)
    print("ADIM 7: Kaydetme")
    print("-" * 50)
    
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/splits", exist_ok=True)
    
    # Train (SMOTE sonrası, scaled) CSV
    train_df = pd.DataFrame(X_train_bal, columns=feature_names)
    train_df['fatigue_class'] = y_train_bal
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    print(f"  → {config.TRAIN_DATA_PATH} ({train_df.shape})")
    
    # Val (scaled) CSV
    val_df = pd.DataFrame(X_val, columns=feature_names)
    val_df['fatigue_class'] = y_val
    val_df.to_csv(config.VAL_DATA_PATH, index=False)
    print(f"  → {config.VAL_DATA_PATH} ({val_df.shape})")
    
    # Test (scaled) CSV
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['fatigue_class'] = y_test
    test_df.to_csv(config.TEST_DATA_PATH, index=False)
    print(f"  → {config.TEST_DATA_PATH} ({test_df.shape})")
    
    # Split indeksleri ve scaler kaydet
    split_data = {
        'X_train': X_train_bal,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train_bal,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler
    }
    joblib.dump(split_data, config.SPLIT_INDICES_PATH)
    print(f"  → {config.SPLIT_INDICES_PATH}")
    
    print("  ✓ Tüm veriler kaydedildi")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("=" * 60)
    print("BDFS — ADIM 3: PREPROCESSING")
    print("=" * 60)
    
    # 1. Veri yükle
    df = load_raw_data()
    
    # 2. Median imputation
    df = median_imputation(df)
    
    # 3. IQR capping
    df = iqr_capping(df)
    
    # 4. One-hot encoding
    df = one_hot_encoding(df)
    
    # 5. Stratified split
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = stratified_split(df)
    
    # 6. Scaling (train'e fit, diğerlerine transform)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = apply_scaling(X_train, X_val, X_test)
    
    # 7. SMOTE (sadece train)
    X_train_bal, y_train_bal = apply_smote(X_train_scaled, y_train)
    
    # 8. Kaydet
    save_all(X_train_scaled, X_val_scaled, X_test_scaled,
             y_train, y_val, y_test,
             X_train_bal, y_train_bal, feature_names, scaler)
    
    print("\n" + "=" * 60)
    print("✓ Adım 3 tamamlandı — Preprocessing başarılı!")
    print("=" * 60)
