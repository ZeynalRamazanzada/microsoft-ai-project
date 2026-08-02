"""
BDFS — Sentetik Veri Üretim Modülü
===================================
Bu modül, Behavioral Decision Fatigue Scoring projesi için
sentetik karar yorgunluğu verisi üretir.

Gizli değişken kaskadı (3 seviye):
  L1: cognitive_energy_t     → Anlık, Ornstein-Uhlenbeck süreci
  L2: session_fatigue_trajectory → Oturum eğrisi
  L3: individual_threshold   → Kişisel direnç eşiği

18 gözlemlenebilir feature + label üretilir.
Sonuçta %4 missing value ve %3 outlier enjekte edilir.

Kullanım:
  python src/data_generator.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress
import os
import sys

# Proje kök dizinini path'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def compute_ez_drift(Pc, MRT):
    """
    EZ-Diffusion Model — Drift Rate (v) hesaplama.
    
    Formül: v = sign(Pc - 0.5) * sqrt(logit(Pc) * (2*Pc - 1)) / sqrt(MRT)
    
    Parametreler:
        Pc  : Doğruluk oranı (0.51–0.99 arasında kırpılmış)
        MRT : Ortalama tepki süresi (Mean Response Time)
    
    Dönüş:
        v : Drift rate (bilgi işleme hızı)
    """
    # Logit hesabı: log(p / (1-p))
    logit_Pc = np.log(Pc / (1.0 - Pc))
    # Drift rate formülü
    v = np.sign(Pc - 0.5) * np.sqrt(np.abs(logit_Pc * (2.0 * Pc - 1.0))) / np.sqrt(np.abs(MRT) + 1e-6)
    return np.clip(v, config.DDM_DRIFT_CLIP[0], config.DDM_DRIFT_CLIP[1])


def compute_ez_boundary(Pc, v):
    """
    EZ-Diffusion Model — Boundary Separation (a) hesaplama.
    
    Formül: a = 2 * logit(Pc) / v
    
    Parametreler:
        Pc : Doğruluk oranı
        v  : Drift rate
    
    Dönüş:
        a : Karar sınırı (boundary separation)
    """
    logit_Pc = np.log(Pc / (1.0 - Pc))
    a = 2.0 * logit_Pc / (v + 1e-6)
    return np.clip(a, config.DDM_BOUNDARY_CLIP[0], config.DDM_BOUNDARY_CLIP[1])


def compute_ez_nondecision(MRT, v, a):
    """
    EZ-Diffusion Model — Non-Decision Time (Ter) hesaplama.
    
    Formül: Ter = MRT - (a / (2*v)) * tanh(a*v / 2)
    
    Parametreler:
        MRT : Ortalama tepki süresi
        v   : Drift rate
        a   : Boundary separation
    
    Dönüş:
        Ter : Non-decision time (motor + algı gecikmesi)
    """
    Ter = MRT - (a / (2.0 * v + 1e-6)) * np.tanh(a * v / 2.0)
    return np.clip(Ter, config.DDM_NONDECISION_CLIP[0], config.DDM_NONDECISION_CLIP[1])


def compute_choice_entropy(pr):
    """
    Tercih entropisi hesapla.
    
    H = -sum(p_i * log(p_i)) formülü ile.
    A/B seçim dağılımının entropisi; yorgunlukta artar.
    
    Parametreler:
        pr : Tercih geri dönüş oranı (preference reversal rate)
    
    Dönüş:
        H : Shannon entropisi (0 ile log(2) arasında kırpılmış)
    """
    # pr'yi seçim olasılığına çevir
    p_a = np.clip(pr, 0.01, 0.99)
    p_b = 1.0 - p_a
    H = -(p_a * np.log2(p_a + 1e-10) + p_b * np.log2(p_b + 1e-10))
    return np.clip(H, 0.0, np.log2(2))


def compute_accuracy_decay(pr_history):
    """
    Doğruluk oranının son 10 turda ne kadar düştüğünü hesapla.
    
    Formül: (accuracy_t0 - accuracy_t10) / 10
    
    Parametreler:
        pr_history : Tercih geri dönüş oranları listesi
    
    Dönüş:
        decay_rate : Doğruluk azalma oranı
    """
    if len(pr_history) < 10:
        return 0.0
    # Doğruluk = 1 - tercih geri dönüş oranı
    accuracy = [1.0 - p for p in pr_history]
    acc_early = np.mean(accuracy[-10:-5]) if len(accuracy) >= 10 else np.mean(accuracy[:5])
    acc_late = np.mean(accuracy[-5:])
    return (acc_early - acc_late) / 5.0


def inject_missing_values(df, fraction, rng):
    """
    Eksik değer enjeksiyonu (MCAR mekanizması).
    
    Bernoulli(p=fraction) maskesi ile rastgele hücrelere NaN atar.
    participant_id, trial, fatigue_class ve session_load sütunları hariç.
    
    Parametreler:
        df       : Veri çerçevesi
        fraction : Eksik değer oranı (ör: 0.04)
        rng      : Numpy random generator
    
    Dönüş:
        df : Eksik değerler eklenmiş veri çerçevesi
    """
    df_copy = df.copy()
    # Eksik değer enjekte edilmeyecek sütunlar
    excluded_cols = ['participant_id', 'trial', 'fatigue_class', 'session_load']
    numeric_cols = [col for col in df_copy.columns if col not in excluded_cols]
    
    n_rows = len(df_copy)
    n_missing_per_col = int(n_rows * fraction)
    
    for col in numeric_cols:
        # Her sütun için rastgele indeksler seç
        missing_idx = rng.choice(n_rows, size=n_missing_per_col, replace=False)
        df_copy.loc[df_copy.index[missing_idx], col] = np.nan
    
    total_missing = df_copy[numeric_cols].isnull().sum().sum()
    total_cells = len(numeric_cols) * n_rows
    print(f"  → Toplam eksik hücre: {total_missing} / {total_cells} "
          f"({100.0 * total_missing / total_cells:.2f}%)")
    
    return df_copy


def inject_outliers(df, fraction, rng):
    """
    Aykırı değer enjeksiyonu.
    
    Rastgele seçilen kayıtlardaki feature değerleri 3-5 kat büyütülür.
    participant_id, trial, fatigue_class ve session_load sütunları hariç.
    
    Parametreler:
        df       : Veri çerçevesi
        fraction : Aykırı değer oranı (ör: 0.03)
        rng      : Numpy random generator
    
    Dönüş:
        df : Aykırı değerler eklenmiş veri çerçevesi
    """
    df_copy = df.copy()
    excluded_cols = ['participant_id', 'trial', 'fatigue_class', 'session_load']
    numeric_cols = [col for col in df_copy.columns if col not in excluded_cols]
    
    n_rows = len(df_copy)
    n_outliers = int(n_rows * fraction)
    
    # Aykırı değer enjekte edilecek indeksler (tüm sütunlar için ortak)
    outlier_idx = rng.choice(n_rows, size=n_outliers, replace=False)
    
    for col in numeric_cols:
        # Çarpan: Uniform(3, 5)
        multipliers = rng.uniform(3.0, 5.0, size=n_outliers)
        # Rastgele yön: pozitif veya negatif
        signs = rng.choice([-1, 1], size=n_outliers)
        current_values = df_copy[col].iloc[outlier_idx].values
        df_copy.loc[df_copy.index[outlier_idx], col] = current_values * multipliers * signs
    
    print(f"  → {n_outliers} kayıda aykırı değer enjekte edildi ({100.0 * fraction:.1f}%)")
    
    return df_copy


# ============================================================
# ANA VERİ ÜRETİM FONKSİYONU
# ============================================================

def generate_bdfs_dataset():
    """
    BDFS sentetik veri setini üretir.
    
    İşlem adımları:
        1. Her katılımcı için gizli parametreler üret (Beta dağılımları)
        2. Her tur için Ornstein-Uhlenbeck süreci ile enerji güncelle
        3. 18 gözlemlenebilir feature hesapla
        4. EZ-Diffusion Model parametrelerini türet
        5. Fatigue label belirle
        6. Eksik değer ve aykırı değer enjekte et
    
    Dönüş:
        df : 150.000 × 21 sütunluk pandas DataFrame (bdfs_raw.csv)
    """
    # Tekrarlanabilirlik için seed ayarla
    rng = np.random.default_rng(config.RANDOM_SEED)
    
    print("=" * 60)
    print("BDFS SENTETİK VERİ ÜRETİMİ BAŞLIYOR")
    print("=" * 60)
    print(f"  Katılımcı sayısı : {config.N_PARTICIPANTS}")
    print(f"  Tur sayısı       : {config.N_TRIALS}")
    print(f"  Toplam kayıt     : {config.N_TOTAL}")
    print(f"  Random seed      : {config.RANDOM_SEED}")
    print()
    
    records = []  # Tüm kayıtları tutacak liste
    
    for participant_id in range(config.N_PARTICIPANTS):
        
        # İlerleme göstergesi (her 500 katılımcıda bir)
        if participant_id % 500 == 0:
            print(f"  → Katılımcı {participant_id}/{config.N_PARTICIPANTS} işleniyor...")
        
        # ============ GİZLİ PARAMETRELER ============
        # L3: Kişisel başlangıç enerjisi — Beta(2, 5) dağılımı
        cog_energy_0 = rng.beta(2, 5)
        
        # L3: Kişisel direnç eşiği — Beta(3, 3) dağılımı
        ind_threshold = rng.beta(3, 3)
        
        # Session load level: {0=düşük, 1=orta, 2=yüksek}
        session_load = rng.choice([0, 1, 2], p=config.SESSION_LOAD_PROBS)
        
        # DDM gerçek parametreleri (gizli, gözlemlenemez)
        drift_rate_true = rng.normal(0.3 + 0.2 * cog_energy_0, 0.1)
        boundary_true = rng.normal(1.2 - 0.3 * session_load, 0.05)
        nondec_true = rng.normal(0.15 + 0.05 * (1 - cog_energy_0), 0.02)
        
        # Anlık enerji değeri (L1)
        cog_energy = cog_energy_0
        
        # Geçmiş değerleri tutacak listeler
        rt_history = []    # Tepki süresi geçmişi
        pr_history = []    # Tercih geri dönüş oranı geçmişi
        
        for trial in range(config.N_TRIALS):
            
            # ============ ENERJİ GÜNCELLEME (Basitleştirilmiş OU Süreci) ============
            # dE_t = -θ * (E_t - μ) * dt + σ * dW_t
            # Basitleştirilmiş: E_{t+1} = E_t - decay_rate + noise
            decay_rate = config.BASE_DECAY * (1 + session_load * 0.5)
            noise = rng.normal(0, config.DECAY_NOISE_STD)
            cog_energy = max(0.0, cog_energy - decay_rate + noise)
            
            # Yorgunluk seviyesi (0=dinç, 1=çok yorgun)
            fatigue_lvl = 1.0 - cog_energy
            
            # ============ TEPKİ SÜRESİ ÜRET ============
            # Lognormal dağılım + yorgunluk etkisi
            rt_mu = np.log(0.6 + 0.3 * fatigue_lvl)
            rt = rng.lognormal(rt_mu, 0.3)
            
            # Gürültü ekle
            rt = rt + rng.normal(0, config.NOISE_LEVEL * rt)
            rt = max(0.1, rt)  # Minimum tepki süresi 100ms
            rt_history.append(rt)
            
            # ============ TERCİH TUTARSIZLIĞI ÜRET ============
            # Yorgunluk arttıkça tercih geri dönüşü artar
            pr = 0.05 + 0.45 * fatigue_lvl + rng.normal(0, 0.03)
            pr = np.clip(pr, 0.0, 1.0)
            pr_history.append(pr)
            
            # Tipping point nonlinearity
            # Yorgunluk %70'in üzerindeyse ani artış
            pr_display = pr * 1.4 if fatigue_lvl > 0.7 else pr
            pr_display = np.clip(pr_display, 0.0, 1.0)
            
            # ============ FEATURE HESAPLAMALARI ============
            
            # --- Grup 1: Tepki Süresi Sinyalleri ---
            window_rt = rt_history[-10:] if len(rt_history) >= 10 else rt_history
            
            # F1: rt_mean — Son 10 turun ortalama tepki süresi
            rt_mean = np.mean(window_rt)
            
            # F2: rt_std — Son 10 turun tepki süresi standart sapması
            rt_std = np.std(window_rt) if len(window_rt) > 1 else 0.0
            
            # F3: rt_slope — Son 10 turun lineer regresyon eğimi
            if len(rt_history) >= 10:
                slope_result = linregress(range(len(window_rt)), window_rt)
                rt_slope = slope_result.slope
            else:
                rt_slope = 0.0
            
            # F4: rt_skew — Tüm geçmiş rt değerlerinin çarpıklığı
            if len(rt_history) >= 3:
                rt_skew = float(stats.skew(rt_history))
            else:
                rt_skew = 0.0
            
            # F5: rt_kurtosis — Tüm geçmiş rt değerlerinin basıklığı
            if len(rt_history) >= 4:
                rt_kurtosis = float(stats.kurtosis(rt_history))
            else:
                rt_kurtosis = 0.0
            
            # Nonlinear etki: Oturum sonunda rt artışı hızlanır
            rt_mean_final = rt_mean + 0.3 * (trial / config.N_TRIALS) ** 2
            
            # --- Grup 2: Tutarsızlık Sinyalleri ---
            
            # F6: pref_reversal_rate — Tercih geri dönüş oranı
            pref_reversal_rate = pr_display
            
            # F7: rolling_incon_5 — Son 5 turun rolling ortalaması
            rolling_incon_5 = np.mean(pr_history[-5:]) if len(pr_history) >= 5 else np.mean(pr_history)
            
            # F8: rolling_incon_10 — Son 10 turun rolling ortalaması
            rolling_incon_10 = np.mean(pr_history[-10:]) if len(pr_history) >= 10 else np.mean(pr_history)
            
            # F9: choice_entropy — Seçim entropisi
            base_entropy = 0.3 + 0.5 * fatigue_lvl + rng.normal(0, 0.05)
            # Etkileşim terimi: oturum sonu + yorgunluk
            sess_pos = trial / config.N_TRIALS
            choice_entropy = base_entropy + 0.2 * sess_pos * fatigue_lvl
            choice_entropy = np.clip(choice_entropy, 0.0, np.log(2))
            
            # --- Grup 3: Temporal Sinyaller ---
            
            # F10: session_position — Normalize tur numarası
            session_position = trial / config.N_TRIALS
            
            # F11: fatigue_slope — Son 15 turun eğimi
            if len(pr_history) >= 15:
                slope_result = linregress(range(15), pr_history[-15:])
                fatigue_slope = slope_result.slope
            else:
                fatigue_slope = 0.0
            
            # F12: accuracy_decay_rate — Doğruluk azalma oranı
            accuracy_decay_rate = compute_accuracy_decay(pr_history)
            
            # F13: inter_trial_variability — Ardışık rt farkları std
            if len(rt_history) >= 10:
                rt_diffs = np.diff(rt_history[-10:])
                inter_trial_var = np.std(rt_diffs)
            else:
                inter_trial_var = 0.0
            
            # --- Grup 4: DDM-Türevli Gizli Özellikler ---
            
            # Doğruluk oranını hesapla (1 - pref_reversal)
            Pc = np.clip(1.0 - pr_display, 0.51, 0.99)
            MRT = rt_mean_final
            
            # F14: ez_drift_rate — EZ-diffusion drift rate
            ez_v = compute_ez_drift(Pc, MRT)
            
            # F15: ez_boundary — EZ-diffusion boundary separation
            ez_a = compute_ez_boundary(Pc, ez_v)
            
            # F16: ez_nondecision — EZ-diffusion non-decision time
            ez_Ter = compute_ez_nondecision(MRT, ez_v, ez_a)
            
            # F17: drift_boundary_ratio — DDM parametre oranı
            db_ratio = ez_v / (ez_a + 1e-6)
            
            # --- Grup 5: Bağlam ---
            
            # F18: task_complexity_level — Görev karmaşıklık seviyesi
            # session_load_level'den türetilir + küçük pertürbasyon
            task_complexity = session_load  # 0, 1, veya 2
            
            # ============ LABEL BELİRLE ============
            # E_t < individual_threshold * FATIGUE_THRESHOLD_MULTIPLIER → 1 (fatigued)
            label = 1 if cog_energy < ind_threshold * config.FATIGUE_THRESHOLD_MULTIPLIER else 0
            
            # ============ KAYDI OLUŞTUR ============
            records.append({
                'participant_id': participant_id,
                'trial': trial,
                'session_load': session_load,
                'rt_mean': rt_mean_final,
                'rt_std': rt_std,
                'rt_slope': rt_slope,
                'rt_skew': rt_skew,
                'rt_kurtosis': rt_kurtosis,
                'pref_reversal_rate': pref_reversal_rate,
                'rolling_incon_5': rolling_incon_5,
                'rolling_incon_10': rolling_incon_10,
                'choice_entropy': choice_entropy,
                'session_position': session_position,
                'fatigue_slope': fatigue_slope,
                'accuracy_decay_rate': accuracy_decay_rate,
                'inter_trial_variability': inter_trial_var,
                'ez_drift_rate': ez_v,
                'ez_boundary': ez_a,
                'ez_nondecision': ez_Ter,
                'drift_boundary_ratio': db_ratio,
                'task_complexity_level': task_complexity,
                'fatigue_class': label
            })
    
    # DataFrame oluştur
    df = pd.DataFrame(records)
    
    print(f"\n  → Ham veri oluşturuldu: {df.shape[0]} satır × {df.shape[1]} sütun")
    
    # ============ CLASS DAĞILIMI KONTROLÜ ============
    pos_ratio = df['fatigue_class'].mean()
    print(f"\n  → Sınıf dağılımı:")
    print(f"    Non-fatigued (0): {(df['fatigue_class'] == 0).sum()} ({100*(1-pos_ratio):.1f}%)")
    print(f"    Fatigued     (1): {(df['fatigue_class'] == 1).sum()} ({100*pos_ratio:.1f}%)")
    
    # Hedef oran kontrolü (%35 ± %3)
    if abs(pos_ratio - config.CLASS_RATIO) > 0.03:
        print(f"\n  ⚠ UYARI: Pozitif sınıf oranı ({pos_ratio:.3f}) hedeften ({config.CLASS_RATIO}) "
              f">{3}% sapma gösteriyor!")
        print(f"  → FATIGUE_THRESHOLD_MULTIPLIER ayarlanması gerekebilir.")
    else:
        print(f"\n  ✓ Sınıf dağılımı hedef aralıkta ({config.CLASS_RATIO} ± 0.03)")
    
    # ============ EKSİK DEĞER ENJEKSİYONU ============
    print(f"\n  Eksik değer enjeksiyonu (MCAR, oran={config.MISSING_FRACTION}):")
    df = inject_missing_values(df, config.MISSING_FRACTION, rng)
    
    # ============ AYKIRI DEĞER ENJEKSİYONU ============
    print(f"\n  Aykırı değer enjeksiyonu (oran={config.OUTLIER_FRACTION}):")
    df = inject_outliers(df, config.OUTLIER_FRACTION, rng)
    
    return df


# ============================================================
# MAIN — Veri üretimi ve kaydetme
# ============================================================

if __name__ == "__main__":
    # Çalışma dizinini proje kök dizinine ayarla
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Veri üret
    df = generate_bdfs_dataset()
    
    # Kaydet
    output_path = config.RAW_DATA_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"VERİ KAYDEDİLDİ: {output_path}")
    print(f"{'='*60}")
    
    # Özet istatistikler
    print("\n" + "=" * 60)
    print("ÖZET İSTATİSTİKLER")
    print("=" * 60)
    print(f"\nBoyut: {df.shape}")
    print(f"\nSütunlar ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        print(f"  {i:2d}. {col:<30s} | dtype: {str(dtype):<10s} | missing: {missing}")
    
    print(f"\nTemel İstatistikler:")
    print(df.describe().round(4).to_string())
    
    print(f"\n✓ Adım 1 tamamlandı — Sentetik veri üretimi başarılı!")
