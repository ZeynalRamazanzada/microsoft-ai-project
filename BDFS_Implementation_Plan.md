# BDFS: Behavioral Decision Fatigue Scoring
## Tam Implementation Planı — Uçtan Uca ML Pipeline

**Proje Adı:** BDFS — A Behavioral Decision Fatigue Scoring Framework Using Sensor-Free Multi-Signal Machine Learning on Synthetic Sequential Choice Data

**Versiyon:** 1.0  
**Framework:** CRISP-DM  
**Reproducibility:** `random_state=42` tüm pipeline boyunca sabit  
**Hedef:** TÜBİTAK 2209-A + Akademik Makale (Frontiers in Cognitive Science / IEEE Access)

---

## İçindekiler

1. [Proje Genel Bakış](#1-proje-genel-bakış)
2. [Ortam Kurulumu](#2-ortam-kurulumu)
3. [Problem Tasarımı (CRISP-DM: Business Understanding)](#3-problem-tasarımı)
4. [Sentetik Veri Üretimi (CRISP-DM: Data Understanding)](#4-sentetik-veri-üretimi)
5. [Veri Ön İşleme (CRISP-DM: Data Preparation — Part 1)](#5-veri-ön-i̇şleme)
6. [Feature Engineering (CRISP-DM: Data Preparation — Part 2)](#6-feature-engineering)
7. [Modelleme (CRISP-DM: Modeling)](#7-modelleme)
8. [Evaluation (CRISP-DM: Evaluation)](#8-evaluation)
9. [Ablation Study](#9-ablation-study)
10. [SHAP Interpretability](#10-shap-interpretability)
11. [Sonuçlar ve Bulgular Tasarımı](#11-sonuçlar-ve-bulgular)
12. [Discussion & Conclusion Taslağı](#12-discussion--conclusion)
13. [Literatür Listesi (APA 7)](#13-literatür-listesi)
14. [Dosya Yapısı](#14-dosya-yapısı)
15. [Claude'a Adım Adım Prompt Sırası](#15-claudea-adım-adım-prompt-sırası)

---

## 1. Proje Genel Bakış

### 1.1 Araştırma Sorusu (Research Question)

> **RQ1:** Fizyolojik sensör kullanmaksızın, yalnızca gözlemlenebilir karar davranış örüntülerinden (tepki süresi dağılımları, tercih tutarsızlığı, temporal düşüş eğrileri) bilişsel yük durumu hangi doğrulukla tahmin edilebilir?
>
> **RQ2:** Drift Diffusion Model (DDM) parametrelerinden türetilen gizli özellikler, ham davranışsal özelliklere kıyasla ne kadar ek tahmin gücü sağlar?

### 1.2 Özgünlük Ekseni (Novelty Statement)

Mevcut literatür iki ayrı sütunda duruyor:

- **Sütun A:** EEG/ECG/fNIRS + Derin Öğrenme → Bilişsel yük tahmini (sensör zorunlu)
- **Sütun B:** Decision fatigue → Gözlemsel sosyal bilim çalışmaları (ML yok)

Bu proje, **Sütun A ile Sütun B arasındaki boşluğu** kapatır:  
Sensörsüz + saf davranışsal sinyal + 5 klasik ML modeli + DDM-türevli gizli özellikler + sentetik veri = **literatürde karşılığı olmayan kombinasyon**.

### 1.3 Veri Özeti

| Parametre | Değer |
|---|---|
| Katılımcı sayısı (sentetik) | 3.000 |
| Tur sayısı / katılımcı | 50 |
| Toplam kayıt | 150.000 satır |
| Feature sayısı (ham) | 18 |
| Feature sayısı (engineering sonrası) | 24 |
| Label | Binary (0 = non-fatigued, 1 = fatigued) |
| Class imbalance | ~65% negatif / 35% pozitif |
| Random seed | 42 |

### 1.4 Model Listesi

| # | Model | Kütüphane |
|---|---|---|
| M1 | Logistic Regression | sklearn |
| M2 | Random Forest | sklearn |
| M3 | XGBoost | xgboost |
| M4 | Support Vector Machine (RBF) | sklearn |
| M5 | K-Nearest Neighbors | sklearn |

---

## 2. Ortam Kurulumu

### 2.1 Gerekli Kütüphaneler

```
numpy==1.26.4
pandas==2.2.1
scipy==1.13.0
scikit-learn==1.4.2
xgboost==2.0.3
shap==0.45.0
matplotlib==3.8.4
seaborn==0.13.2
imbalanced-learn==0.12.2
joblib==1.4.0
```

### 2.2 Kurulum Komutu

```bash
pip install numpy pandas scipy scikit-learn xgboost shap matplotlib seaborn imbalanced-learn joblib
```

### 2.3 Config Dosyası — `config.py`

Tüm sabitler tek dosyada toplanır. Pipeline boyunca bu dosyadan çağrılır.

```python
# config.py
RANDOM_SEED = 42
N_PARTICIPANTS = 3000
N_TRIALS = 50
N_TOTAL = N_PARTICIPANTS * N_TRIALS          # 150_000
CLASS_RATIO = 0.35                           # pozitif sınıf oranı
TEST_SIZE = 0.15
VAL_SIZE = 0.15
CV_FOLDS = 10
NOISE_LEVEL = 0.08
OUTLIER_FRACTION = 0.03
MISSING_FRACTION = 0.04
FATIGUE_THRESHOLD_MULTIPLIER = 0.4
```

---

## 3. Problem Tasarımı

### 3.1 Problem Modeli

Her "katılımcı", bir oturumda ardışık 50 karar tur geçirir. Her turda şunlar simüle edilir:

- İki seçenek arasında tercih yapması (A veya B)
- Tepki süresi (reaction time) üretmesi
- Tercih tutarlılığı (aynı çift yeniden geldiğinde aynı seçimi yapıp yapmadığı)

Katılımcının **gizli bilişsel enerjisi** oturum boyunca düşer. Enerji belirli bir kişisel eşiğin altına indiğinde katılımcı "fatigued" olarak etiketlenir.

### 3.2 Label Üretim Mantığı

```
ADIM 1: Her katılımcı için gizli parametreler üret:
    - cognitive_energy_0   ← Beta(2, 5)       [başlangıç enerjisi, 0-1 arası]
    - individual_threshold ← Beta(3, 3)       [kişisel direnç eşiği, 0-1 arası]
    - session_load_level   ← {0, 1, 2}        [düşük/orta/yüksek yük, 20/50/30]

ADIM 2: Her tur için enerji güncelle (Ornstein-Uhlenbeck süreci):
    dE_t = -θ * (E_t - μ) * dt + σ * dW_t
    [θ=0.3, μ=0.2, σ=0.05, dt=1/50]
    Basitleştirilmiş: E_{t+1} = E_t - decay_rate + noise
    decay_rate = 0.008 * (1 + session_load_level * 0.5)
    noise      = Normal(0, 0.02)

ADIM 3: Fatigue label belirle:
    E_t < individual_threshold * FATIGUE_THRESHOLD_MULTIPLIER → label = 1
    Aksi halde → label = 0

ADIM 4: İmbalance kontrolü:
    Hedef: ~%35 pozitif
    Gerçek oran kontrol edilir, sapma >%3 ise threshold multiplier ayarlanır
```

### 3.3 Neden Bu Label Mantığı?

Bu yaklaşım "ground truth without sensors" oluşturur. Gerçek dünyada EEG ile ölçülen şeyi, burada matematiksel bir gizli süreçle simüle ediyoruz. Bu yöntemin akademik gerekçesi: sintetik veri ile kontrollü deney yapma, replicability sağlar ve confound variable kontrolü mümkündür.

---

## 4. Sentetik Veri Üretimi

### 4.1 Gizli Değişken Kaskadı (3 Seviye)

```
L1: cognitive_energy_t     [anlık, Ornstein-Uhlenbeck süreci]
    └─ Her tur güncellenir
    └─ Gözlemlenemeyen

L2: session_fatigue_trajectory  [oturum eğrisi, sigmoid decay]
    └─ E_t değerlerinin oturum ortalaması
    └─ Gözlemlenemeyen

L3: individual_threshold        [kişisel direnç, Beta prior]
    └─ Sabit, katılımcıya özgü
    └─ Gözlemlenemeyen

→ Üçü de hiçbir zaman doğrudan feature olarak verilmez.
→ Sadece gözlemlenebilir sinyalleri üretmek için kullanılır.
```

### 4.2 EZ-Diffusion Model Parametreleri

DDM teorisinde karar süreci üç parametreyle açıklanır. Biz bunları sentetik olarak üretip feature olarak kullanacağız.

```
drift_rate_v     = 0.5 * cognitive_energy_t + Normal(0, 0.1)
    [bilgi işleme hızı; enerji düştükçe drift rate düşer]
    Dağılım: Normal(0.3, 0.15), clip [0.05, 1.0]

boundary_a       = 1.2 - 0.3 * session_load_level + Normal(0, 0.05)
    [karar temkinliliği; yüksek yük → düşük boundary]
    Dağılım: Normal(1.0, 0.2), clip [0.5, 2.0]

nondecision_Ter  = 0.15 + 0.08 * (1 - cognitive_energy_t) + Normal(0, 0.02)
    [motor + algı gecikmesi; enerji düştükçe artar]
    Dağılım: Normal(0.2, 0.05), clip [0.05, 0.5]
```

### 4.3 18 Gözlemlenebilir Feature — Tam Üretim Planı

#### Grup 1: Tepki Süresi Sinyalleri (5 feature)

```
rt_mean:
    Üretim: Lognormal(mu=log(0.6), sigma=0.3) * (1 + 0.4 * fatigue_level)
    [yorgunluk arttıkça tepki süresi uzar]
    Gürültü: +/- Uniform(0, 0.05)

rt_std:
    Üretim: rt_mean * Beta(2, 5) * (1 + 0.3 * fatigue_level)
    [yorgunluk arttıkça varyans artar]

rt_skew:
    Üretim: scipy.stats.skewnorm(a=3) * fatigue_level
    [yorgunlukta dağılım sağa çarpıklaşır]

rt_kurtosis:
    Üretim: Normal(3, 0.5) + 1.2 * fatigue_level
    [yorgunlukta aykırı değerler artar → kurtosis yükselir]

rt_slope:
    Üretim: Son 10 turun rt değerlerinin lineer regresyon eğimi
    [pozitif eğim = yavaşlama = yorgunluk sinyali]
    rt_slope = linregress(range(10), rt_window[-10:]).slope
```

#### Grup 2: Tutarsızlık Sinyalleri (4 feature)

```
pref_reversal_rate:
    Üretim: 0.05 + 0.45 * fatigue_level + Normal(0, 0.03)
    [yorgunluk → daha fazla tercih geri dönüşü]
    clip [0, 1]

rolling_incon_5:
    Son 5 turun pref_reversal oranının rolling ortalaması
    [kısa pencere tutarsızlık sinyali]

rolling_incon_10:
    Son 10 turun pref_reversal oranının rolling ortalaması
    [uzun pencere tutarsızlık sinyali]

choice_entropy:
    H = -sum(p_i * log(p_i)) üzerinden hesaplanır
    [A/B seçim dağılımının entropisi; yorgunlukta artar]
    Üretim: 0.3 + 0.5 * fatigue_level + Normal(0, 0.05)
    clip [0, log(2)]
```

#### Grup 3: Temporal Sinyaller (4 feature)

```
session_position:
    Tur numarası / 50
    [0.0 = başlangıç, 1.0 = bitiş]
    Normalize edilmiş, gürültüsüz

fatigue_slope:
    Son 15 turun pref_reversal_rate değerlerinin lineer eğimi
    [pozitif eğim = artan yorgunluk trendi]

accuracy_decay_rate:
    Doğru tercih oranının son 10 turda ne kadar düştüğü
    Üretim: (accuracy_t0 - accuracy_t10) / 10
    [yorgunlukla doğru karar kalitesi düşer]

inter_trial_variability:
    Ardışık rt değerleri arasındaki farkların standart sapması
    std(diff(rt_series[-10:]))
    [yorgunlukta tutarsız rt örüntüsü]
```

#### Grup 4: DDM-Türevli Gizli Özellikler (4 feature)

```
ez_drift_rate:
    EZ-diffusion formülünden hesaplanır:
    v = sign(Pc - 0.5) * (logit(Pc) * (2*Pc - 1))^0.5 / sqrt(MRT)
    Burada Pc = doğruluk oranı, MRT = ortalama rt

ez_boundary:
    a = 2 * logit(Pc) / v
    [karar sınırı tahmini]

ez_nondecision:
    Ter = MRT - (a / (2*v)) * tanh(a*v / 2)
    [motor + algı gecikmesi tahmini]

drift_boundary_ratio:
    ez_drift_rate / ez_boundary
    [düşük oran → yavaş ve temkinli → yorgunluk sinyali]
```

#### Grup 5: Bağlam (1 feature)

```
task_complexity_level:
    Kategorik: {0, 1, 2} → One-hot encoding sonrası 3 binary feature
    session_load_level değerinden üretilir + küçük rassal pertürbasyon
    [confound variable olarak kontrol edilmesi gerekir]
```

### 4.4 Korelasyon Yapısı

Aşağıdaki feature çiftleri kasıtlı olarak korele üretilir:

```
rt_mean ↔ rt_std:             Pearson r ≈ 0.65  [yüksek korelasyon, beklenen]
ez_drift_rate ↔ rt_slope:     Pearson r ≈ -0.55 [negatif korelasyon]
pref_reversal ↔ rolling_incon: Pearson r ≈ 0.75 [yüksek, tutarsızlık grubu]
session_position ↔ rt_slope:  Pearson r ≈ 0.40  [orta korelasyon]
ez_boundary ↔ rt_kurtosis:    Pearson r ≈ 0.30  [düşük korelasyon]
```

Üretim yöntemi: Bu korelasyonu oluşturmak için `numpy.random.multivariate_normal` ile kovaryans matrisi tanımlanır. Her özellik grubu için kovaryans bloğu ayrı belirlenir, bloklar birleştirilir.

### 4.5 Nonlinear İlişkiler

Sadece lineer ilişkiler gerçekçi değil. Şu nonlinear yapılar eklenir:

```
rt_mean üzerinde karesel etki:
    rt_final = rt_mean + 0.3 * session_position^2
    [oturum sonunda rt artışı hızlanır]

pref_reversal üzerinde eşik etkisi:
    Eğer fatigue_level > 0.7:
        pref_reversal = pref_reversal * 1.4   [ani artış]
    [yorgunluğun "tipping point" davranışı]

choice_entropy üzerinde etkileşim terimi:
    choice_entropy = choice_entropy + 0.2 * session_position * fatigue_level
    [iki değişkenin çarpım etkisi]
```

### 4.6 Noise Injection

```
Genel gürültü (tüm numerik feature'lara):
    noise = Normal(0, NOISE_LEVEL) * feature_std
    feature = feature + noise

Outlier enjeksiyonu (%3 kayıt):
    outlier_idx = random.choice(total_rows, size=N*0.03)
    feature[outlier_idx] = feature[outlier_idx] * Uniform(3, 5)
    [aykırı değer oluşturur — gerçekçi veri kalitesi simülasyonu]

Missing value simülasyonu (%4 kayıt, MCAR mekanizması):
    missing_mask = Bernoulli(p=0.04)
    feature[missing_mask] = NaN
    [Completely At Random — basit imputation geçerlidir]
```

### 4.7 Pseudo-code — Tam Veri Üretim Döngüsü

```
FUNCTION generate_bdfs_dataset(config):

    SET seed(42)
    INIT records = []

    FOR participant_id IN range(N_PARTICIPANTS):

        # Gizli parametreler
        cog_energy_0      = Beta(2, 5)
        ind_threshold     = Beta(3, 3)
        session_load      = choice([0,1,2], p=[0.20,0.50,0.30])
        drift_rate_true   = Normal(0.3 + 0.2*cog_energy_0, 0.1)
        boundary_true     = Normal(1.2 - 0.3*session_load, 0.05)
        nondec_true       = Normal(0.15 + 0.05*(1-cog_energy_0), 0.02)

        cog_energy = cog_energy_0
        rt_history = []
        pr_history = []

        FOR trial IN range(N_TRIALS):

            # Enerji güncelleme (basitleştirilmiş OU süreci)
            decay       = 0.008 * (1 + session_load * 0.5)
            cog_energy  = max(0, cog_energy - decay + Normal(0, 0.02))
            fatigue_lvl = 1 - cog_energy

            # Tepki süresi üret
            rt = Lognormal(log(0.6 + 0.3*fatigue_lvl), 0.3)
            rt = rt + Normal(0, NOISE_LEVEL * rt)
            rt_history.append(rt)

            # Tercih tutarsızlığı üret
            pr = clip(0.05 + 0.45*fatigue_lvl + Normal(0, 0.03), 0, 1)
            pr_history.append(pr)

            # Tipping point nonlinearity
            IF fatigue_lvl > 0.7:
                pr = pr * 1.4

            # Feature hesapla
            rt_mean   = mean(rt_history[-10:]) IF len>=10 ELSE mean(rt_history)
            rt_std    = std(rt_history[-10:])
            rt_slope  = linregress(rt_history[-10:]).slope IF len>=10 ELSE 0
            roll5     = mean(pr_history[-5:])
            roll10    = mean(pr_history[-10:])
            sess_pos  = trial / N_TRIALS
            fat_slope = linregress(pr_history[-15:]).slope IF len>=15 ELSE 0

            # EZ-diffusion hesapla
            Pc  = clip(1 - pr, 0.51, 0.99)
            MRT = rt_mean
            ez_v   = compute_ez_drift(Pc, MRT)
            ez_a   = compute_ez_boundary(Pc, ez_v)
            ez_Ter = compute_ez_nondecision(MRT, ez_v, ez_a)
            db_ratio = ez_v / (ez_a + 1e-6)

            # Label
            label = 1 IF cog_energy < ind_threshold * FATIGUE_THRESHOLD_MULTIPLIER ELSE 0

            # Kayıt
            records.append({
                'participant_id': participant_id,
                'trial': trial,
                'session_load': session_load,
                'rt_mean': rt_mean, 'rt_std': rt_std,
                'rt_slope': rt_slope,
                'rt_skew': skew(rt_history), 'rt_kurtosis': kurtosis(rt_history),
                'pref_reversal_rate': pr,
                'rolling_incon_5': roll5, 'rolling_incon_10': roll10,
                'choice_entropy': entropy_calc(pr),
                'session_position': sess_pos, 'fatigue_slope': fat_slope,
                'accuracy_decay_rate': accuracy_decay(pr_history),
                'inter_trial_variability': std(diff(rt_history[-10:])),
                'ez_drift_rate': ez_v, 'ez_boundary': ez_a,
                'ez_nondecision': ez_Ter, 'drift_boundary_ratio': db_ratio,
                'fatigue_class': label
            })

    df = DataFrame(records)
    df = inject_missing(df, fraction=MISSING_FRACTION)
    df = inject_outliers(df, fraction=OUTLIER_FRACTION)
    RETURN df
```

---

## 5. Veri Ön İşleme

### 5.1 Genel Akış

```
Ham veri (150.000 × 21 sütun)
    ↓
Missing value analiz + imputation
    ↓
Outlier detection + handling
    ↓
Scaling (StandardScaler)
    ↓
Encoding (task_complexity_level → one-hot)
    ↓
Temiz veri (150.000 × 24 sütun)
```

### 5.2 Missing Value Handling

```
ADIM 1: Her sütunun eksik değer oranını hesapla
    missing_report = df.isnull().sum() / len(df)

ADIM 2: Eksik değer mekanizması kontrol et
    Hedef: MCAR (completely at random) → basit imputation yeterli
    Test: Little's MCAR test (statsmodels)

ADIM 3: İmputation stratejisi
    Numerik feature'lar: median imputation
        [outlier'a karşı dirençli, mean'den daha güvenli]
    Kategorik feature'lar: mode imputation
    Araç: sklearn.impute.SimpleImputer(strategy='median')

ADIM 4: Kontrol
    Imputation sonrası missing_report tekrar çalıştırılır → tümü 0 olmalı
```

### 5.3 Outlier Detection ve Handling

```
ADIM 1: IQR yöntemi ile tespit
    Q1 = percentile(25)
    Q3 = percentile(75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_mask = (feature < lower) | (feature > upper)

ADIM 2: Z-score yöntemi ile doğrulama
    z = (feature - mean) / std
    z_outlier_mask = abs(z) > 3.0

ADIM 3: Her iki yöntemde de outlier olan kayıtları işaretle

ADIM 4: Handling stratejisi
    → Capping (Winsorization): clip to [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    → Silme değil! Çünkü %3 outlier kasıtlı enjekte edildi;
      gerçek verinin simülasyonu olarak korunur
    → Notasyon: kaçının outlier olduğu raporlanır (Discussion'da kullanılacak)
```

### 5.4 Feature Scaling

```
StandardScaler kullanılacak (Z-score normalization):
    X_scaled = (X - mean) / std

Neden StandardScaler?
    → SVM ve KNN mesafe tabanlı → scaling zorunlu
    → Logistic Regression gradient descent'te scaling yardımcı
    → Tree-based modeller (RF, XGB) scaling'e duyarsız
      ama tutarlılık için uygulanır

Kritik kural: Scaler SADECE train set üzerinde fit edilir.
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled   = scaler.transform(X_val)      ← transform, fit değil!
    X_test_scaled  = scaler.transform(X_test)     ← transform, fit değil!

MinMaxScaler alternatifi:
    KNN için MinMaxScaler denenebilir (ablation'da)
    Ama genel pipeline StandardScaler ile gider
```

### 5.5 Encoding

```
task_complexity_level: {0, 1, 2} → One-Hot Encoding
    sklearn.preprocessing.OneHotEncoder(drop='first', sparse=False)
    [drop='first' ile dummy variable trap engellenir]
    Çıktı: complexity_medium, complexity_high (2 binary sütun)

participant_id: modele verilmez (ID sütunu, drop edilir)
trial: modele verilmez (ham, session_position olarak zaten var)
```

### 5.6 Train / Validation / Test Split

```
Toplam: 150.000 kayıt

ADIM 1: Test set ayır (%15)
    X_test, y_test = 22.500 kayıt
    train_temp = 127.500 kayıt

ADIM 2: Validation set ayır (%15 orijinal = %17.6 temp'ten)
    X_val, y_val = 22.500 kayıt
    X_train, y_train = 105.000 kayıt

Kritik: train/val/test split'te stratify=y kullanılır
    → Class imbalance korunur her sette
    sklearn.model_selection.train_test_split(stratify=y)

ADIM 3: Class imbalance stratejisi (eğitim seti üzerinde)
    SMOTE (Synthetic Minority Over-sampling Technique)
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    NOT: SMOTE sadece X_train'e uygulanır, validation ve test'e değil!
    Bu data leakage'ı engeller.
```

---

## 6. Feature Engineering

### 6.1 Correlation Analizi

```
ADIM 1: Pearson korelasyon matrisi (numerik feature'lar)
    corr_matrix = X_train.corr(method='pearson')
    Görselleştirme: seaborn.heatmap(annot=True, fmt='.2f')

ADIM 2: Yüksek korelasyon tespiti
    high_corr_pairs = [(f1, f2) for |r| > 0.80]
    Beklenen yüksek korelasyonlar:
        - pref_reversal_rate ↔ rolling_incon_10 (r ≈ 0.75)
        - rt_mean ↔ rt_std (r ≈ 0.65)
    Bu çiftler için: birini drop etmek yerine PCA ile boyut azaltma denenebilir

ADIM 3: VIF (Variance Inflation Factor) hesapla
    Multicollinearity tespiti için
    VIF > 10 → problematik
    statsmodels.stats.outliers_influence.variance_inflation_factor
```

### 6.2 Mutual Information Analizi

```
MI her feature'ın label ile ne kadar bağımsız bilgi içerdiğini ölçer.
    from sklearn.feature_selection import mutual_info_classif
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

Beklenen yüksek MI feature'ları:
    1. ez_drift_rate           (DDM gizli bilgi)
    2. pref_reversal_rate      (doğrudan yorgunluk sinyali)
    3. fatigue_slope           (temporal trend)
    4. drift_boundary_ratio    (DDM oranı)
    5. rolling_incon_10        (uzun pencere)

Görselleştirme: Yatay bar chart, azalan sıraya göre
```

### 6.3 Feature Selection Stratejisi

```
Strateji: 3 yöntem kombine edilir

Yöntem 1: MI tabanlı → İlk 15 feature
Yöntem 2: RF feature importance → İlk 15 feature
Yöntem 3: Recursive Feature Elimination (RFE) → 15 feature
    from sklearn.feature_selection import RFE

Seçim mantığı:
    2+ yöntemde top-15'e giren feature → "core feature"
    Core feature'lar her modelde kullanılır

Ablation için:
    Full set (18 feature) vs Core set (12-15 feature) karşılaştırması

PCA (opsiyonel):
    Sadece eğer multicollinearity ciddi ise uygula
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95, random_state=42)
    [varyansın %95'ini açıklayan bileşen sayısı]
    PCA uygulandıysa bu ayrı bir "PCA pipeline" olarak raporlanır
```

### 6.4 Yeni Türetilmiş Feature'lar (Engineering)

```
Eklenen 6 yeni feature (18'den 24'e çıkar):

1. rt_cv (Coefficient of Variation):
    rt_cv = rt_std / rt_mean
    [normalize varyasyon; yorgunlukla artar]

2. incon_acceleration:
    incon_acceleration = rolling_incon_5 - rolling_incon_10
    [son 5 tur ile son 10 tur tutarsızlığı arasındaki fark]
    [pozitif = hızlanan tutarsızlık = erken uyarı sinyali]

3. ddm_fatigue_index:
    ddm_fatigue_index = (1 / ez_drift_rate) * ez_nondecision
    [DDM parametrelerinden türetilen bileşik yorgunluk skoru]

4. temporal_load_interaction:
    temporal_load_interaction = session_position * task_complexity_level
    [etkileşim terimi; oturum sonu + yüksek yük = en riskli kombinasyon]

5. rt_kurtosis_normalized:
    rt_kurtosis_normalized = (rt_kurtosis - 3) / rt_std
    [excess kurtosis; ağır kuyruk davranışı]

6. decision_efficiency:
    decision_efficiency = (1 - pref_reversal_rate) / rt_mean
    [birim sürede doğru karar oranı; yorgunlukla hızla düşer]
```

---

## 7. Modelleme

### 7.1 Pipeline Yapısı

Tüm modeller sklearn Pipeline ile sarmalanır. Bu data leakage'ı engeller.

```
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(...))
])
```

Her model için ayrı pipeline oluşturulur. GridSearchCV veya RandomizedSearchCV pipeline üzerinde çalıştırılır.

### 7.2 Model 1 — Logistic Regression

**Neden seçildi:** Baseline model; yorgunluğun doğrusal sinyallerini (rt_slope, fatigue_slope) iyi yakalar. Interpretability yüksek.

**Nerede başarısız olur:** Nonlinear etkileşimler (tipping point davranışı, etkileşim terimleri) yakalanmaz. rt_kurtosis gibi yüksek momentler etkisiz kalır.

**Bias-Variance:** Yüksek bias, düşük variance. Underfitting riski vardır.

```
Hyperparameter arama alanı (GridSearchCV):
    'C': [0.01, 0.1, 1, 10, 100]
    'penalty': ['l1', 'l2']
    'solver': ['liblinear', 'saga']   ← l1 için liblinear/saga gerekli
    'class_weight': ['balanced', None]
    'max_iter': [1000]

Toplam kombinasyon: 5 × 2 × 2 × 2 = 40 (bazıları geçersiz, otomatik elenir)
CV: StratifiedKFold(n_splits=5) — hyperparameter tuning için
```

### 7.3 Model 2 — Random Forest

**Neden seçildi:** Nonlinear ilişkileri yakalar, feature importance analizi sağlar, SHAP ile uyumlu. Eksik değerlere ve outlier'a nispeten dirençli.

**Nerede başarısız olur:** DDM feature'larının birbirine korelasyonu overfitting'e yol açabilir. Çok derin ağaçlarda yüksek varyans.

**Bias-Variance:** Düşük bias, yüksek variance (derinlikle artar). Regularization: max_depth, min_samples_leaf.

```
Hyperparameter arama alanı (RandomizedSearchCV, n_iter=50):
    'n_estimators': [100, 200, 300, 500]
    'max_depth': [5, 10, 15, 20, None]
    'min_samples_split': [2, 5, 10, 20]
    'min_samples_leaf': [1, 2, 4, 8]
    'max_features': ['sqrt', 'log2', 0.5]
    'class_weight': ['balanced', 'balanced_subsample', None]

n_iter=50 → 150.000 satır büyük veri, RandomizedSearch hesaplı
CV: StratifiedKFold(n_splits=5)
```

### 7.4 Model 3 — XGBoost

**Neden seçildi:** Gradient boosting, imbalanced data'da güçlü (scale_pos_weight). Regularization built-in (L1/L2). Sequential boosting nonlinear etkileşimleri yakalamada mükemmel.

**Nerede başarısız olur:** Hyperparameter sayısı fazla → tuning süresi uzar. Çok derin boosting aşamaları overfitting yaratabilir.

**Bias-Variance:** İyi tune edildiğinde düşük bias + kontrollü variance. n_estimators + learning_rate trade-off kritik.

```
Hyperparameter arama alanı (RandomizedSearchCV, n_iter=60):
    'n_estimators': [100, 200, 300, 500]
    'max_depth': [3, 4, 5, 6, 8]
    'learning_rate': [0.01, 0.05, 0.1, 0.2]
    'subsample': [0.7, 0.8, 0.9, 1.0]
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    'reg_alpha': [0, 0.1, 1.0]         ← L1
    'reg_lambda': [1, 5, 10]           ← L2
    'scale_pos_weight': [1, 2, 1.86]   ← imbalance için; 1.86 = neg/pos oranı

Early stopping kullanılacak:
    eval_set = [(X_val_scaled, y_val)]
    early_stopping_rounds = 20
    [validation kaybı 20 round'da iyileşmezse dur]
```

### 7.5 Model 4 — SVM (RBF Kernel)

**Neden seçildi:** Yüksek boyutlu uzayda güçlü margin. RBF kernel nonlinear karar sınırları oluşturur.

**Nerede başarısız olur:** 150.000 satır büyük veri için yavaş (O(n²) veya O(n³) karmaşıklık). Scaling zorunlu. Probability calibration ekstra adım gerektirir.

**Bias-Variance:** C parametresi ile kontrol: düşük C → yüksek bias, yüksek C → yüksek variance.

```
ÖNEMLİ: SVM 150.000 satır için çok yavaş olacak.
Çözüm: LinearSVC veya subsampling

Strateji:
    ADIM 1: LinearSVC dene (daha hızlı, lineer karar sınırı)
    ADIM 2: Eğer süre makul ise RBF dene (subsample ile)
    ADIM 3: SVC(probability=True, kernel='rbf') için
            n_samples = 30.000 (subsample %20)
            [Tam 150K satır üzerinde SVM pratik değil]

Hyperparameter arama alanı (GridSearchCV):
    'C': [0.1, 1, 10, 100]
    'gamma': ['scale', 'auto', 0.01, 0.001]
    'class_weight': ['balanced', None]

CalibratedClassifierCV ile probability output eklenir:
    from sklearn.calibration import CalibratedClassifierCV
    svm_calibrated = CalibratedClassifierCV(svm_model, cv=5)
    [ROC-AUC için probability gerekli]
```

### 7.6 Model 5 — K-Nearest Neighbors

**Neden seçildi:** Baseline olarak. Temporal feature'lar için beklenen düşük performans, bu kasıtlı. "Neden KNN başarısız olur?" sorusu akademik tartışmaya değer katar.

**Nerede başarısız olur:** Temporal sinyaller (rt_slope, fatigue_slope) komşuluk mantığıyla uyumsuz. Yüksek boyut → curse of dimensionality. Büyük veri → yavaş prediction.

**Bias-Variance:** Düşük k → düşük bias + yüksek variance. Yüksek k → yüksek bias + düşük variance.

```
Hyperparameter arama alanı (GridSearchCV):
    'n_neighbors': [3, 5, 7, 11, 15, 21]
    'weights': ['uniform', 'distance']
    'metric': ['euclidean', 'manhattan']
    'p': [1, 2]                        ← Minkowski p-norm

Beklenen sonuç: ROC-AUC ≈ 0.70-0.75 (diğerlerinin altında)
Bu beklenti Discussion bölümünde açıklanır.
```

### 7.7 Hyperparameter Tuning Özeti

| Model | Yöntem | n_iter / kombinasyon | CV folds |
|---|---|---|---|
| LR | GridSearchCV | ~40 | 5 |
| RF | RandomizedSearchCV | 50 | 5 |
| XGB | RandomizedSearchCV | 60 | 5 |
| SVM | GridSearchCV (subsample) | 32 | 5 |
| KNN | GridSearchCV | 48 | 5 |

**Süre tahmini (8-core CPU):** ~45-90 dakika toplam (paralel işlem: `n_jobs=-1`)

---

## 8. Evaluation

### 8.1 Stratified K-Fold Cross Validation

```
from sklearn.model_selection import StratifiedKFold, cross_validate

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

cv_results = cross_validate(
    best_model,
    X_train_scaled,
    y_train,
    cv=skf,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

Raporlanacak: mean ± std her metrik için
Örnek format: "ROC-AUC = 0.891 ± 0.012"
```

### 8.2 Hold-out Test Evaluation

```
Her model için ayrı ayrı:

1. Best model (GridSearch/RandomSearch sonrası) X_test üzerinde değerlendir
2. Metrics:
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)   ← y_prob = predict_proba[:,1]

3. Confusion Matrix:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap='Blues')

4. ROC Curve (5 model tek grafikte):
    from sklearn.metrics import roc_curve, auc
    for model in all_models:
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc(fpr,tpr):.3f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
```

### 8.3 Model Karşılaştırma Tablosu

Makalede kullanılacak ana tablo formatı:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV Mean±Std |
|---|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — | — |
| Random Forest | — | — | — | — | — | — |
| XGBoost | — | — | — | — | — | — |
| SVM (RBF) | — | — | — | — | — | — |
| KNN | — | — | — | — | — | — |

Tabloda en iyi değer **kalın**, ikinci en iyi *italik* gösterilir.

### 8.4 İstatistiksel Anlamlılık Testi

```
McNemar Testi — iki model arasındaki fark anlamlı mı?

from mlxtend.evaluate import mcnemar_table, mcnemar

# XGBoost vs Random Forest karşılaştırması (en çok beklenen rakipler)
ct = mcnemar_table(y_target=y_test,
                   y_model1=xgb_pred,
                   y_model2=rf_pred)
chi2, p = mcnemar(ary=ct, corrected=True)

H0: İki modelin performansı arasında fark yoktur
p < 0.05 → H0 reddedilir → fark istatistiksel olarak anlamlıdır

Tüm çift kombinasyonlar (5C2 = 10 karşılaştırma) test edilir
Bonferroni düzeltmesi: α = 0.05 / 10 = 0.005
```

---

## 9. Ablation Study

Bu bölüm projeyi standart ML ödevinden ayıran kritik katmandır.

### 9.1 Ablation Tasarımı

4 farklı feature seti ile her model tekrar eğitilir:

```
Config A — Full Model (24 feature):
    Tüm feature'lar dahil (ham 18 + türetilmiş 6)
    → Referans model

Config B — No DDM Features (20 feature):
    ez_drift_rate, ez_boundary, ez_nondecision, drift_boundary_ratio çıkarılır
    → "DDM feature'ları gerçekten katkı sağlıyor mu?"

Config C — No Temporal Features (20 feature):
    fatigue_slope, session_position, accuracy_decay_rate, rt_slope çıkarılır
    → "Zamansal sinyaller kritik mi?"

Config D — Behavioral Baseline Only (5 feature):
    Sadece: rt_mean, rt_std, pref_reversal_rate, rolling_incon_5, choice_entropy
    → "Minimum sinyalle ne kadar gidilebilir?"
```

### 9.2 Ablation Sonuç Tablosu

| Config | Features | XGB ROC-AUC | RF ROC-AUC | Δ vs Full |
|---|---|---|---|---|
| A (Full) | 24 | — | — | — |
| B (No DDM) | 20 | — | — | -Δ₁ |
| C (No Temporal) | 20 | — | — | -Δ₂ |
| D (Baseline) | 5 | — | — | -Δ₃ |

Beklenti: Δ₁ > Δ₂ > 0, yani DDM feature'ları temporal feature'lardan daha fazla katkı sağlar.

---

## 10. SHAP Interpretability

### 10.1 XGBoost için SHAP Analizi

```
import shap

explainer = shap.TreeExplainer(xgb_best_model)
shap_values = explainer.shap_values(X_test_scaled)

Üretilecek 4 görsel:

1. SHAP Summary Plot (Beeswarm):
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names)
    [Her feature'ın genel etki dağılımı]

2. SHAP Bar Plot (ortalama mutlak SHAP):
    shap.summary_plot(shap_values, X_test_scaled, plot_type='bar')
    [Feature importance sıralaması — RF importance ile karşılaştır]

3. SHAP Dependence Plot (en önemli feature için):
    shap.dependence_plot('ez_drift_rate', shap_values, X_test_scaled)
    [Tek feature'ın label üzerindeki etkisi + etkileşim]

4. SHAP Waterfall Plot (tek örnek için):
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test_scaled[0]
    ))
    [Tek bir "fatigued" katılımcının tahmininin açıklaması]
```

### 10.2 Random Forest Feature Importance Karşılaştırması

```
rf_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_best_model.feature_importances_
}).sort_values('importance', ascending=False)

# SHAP ile karşılaştır
shap_importance = pd.DataFrame({
    'feature': feature_names,
    'shap_mean_abs': np.abs(shap_values).mean(axis=0)
}).sort_values('shap_mean_abs', ascending=False)

# İki sıralamayı yan yana göster
# Spearman rank correlation hesapla → ne kadar tutarlı?
from scipy.stats import spearmanr
rho, p = spearmanr(rf_importance['importance'], shap_importance['shap_mean_abs'])
```

---

## 11. Sonuçlar ve Bulgular Tasarımı

### 11.1 Üretilecek Görseller (Tamamı)

```
Görsel 1: Sentetik veri dağılımları
    - Her feature için histogram (2 sınıf ayrı renkle)
    - 4×5 subplot grid
    - Dosya: figures/01_feature_distributions.png

Görsel 2: Korelasyon matrisi
    - Tüm feature'lar
    - seaborn heatmap, annotated
    - Dosya: figures/02_correlation_matrix.png

Görsel 3: Class imbalance
    - Bar chart: 0 vs 1 sınıf sayısı
    - Dosya: figures/03_class_distribution.png

Görsel 4: Mutual Information skorları
    - Azalan sıra, yatay bar chart
    - Dosya: figures/04_mutual_information.png

Görsel 5: Confusion matrix (5 model, yan yana)
    - 1×5 subplot
    - Dosya: figures/05_confusion_matrices.png

Görsel 6: ROC curves (5 model tek grafikte)
    - AUC değerleri legend'da
    - Dosya: figures/06_roc_curves.png

Görsel 7: Model karşılaştırma (radar chart veya grouped bar)
    - 5 metrik × 5 model
    - Dosya: figures/07_model_comparison.png

Görsel 8: CV box plots
    - Her model için ROC-AUC dağılımı (10 fold)
    - Dosya: figures/08_cv_boxplots.png

Görsel 9: SHAP summary plot
    - Dosya: figures/09_shap_summary.png

Görsel 10: SHAP dependence plot (en önemli feature)
    - Dosya: figures/10_shap_dependence.png

Görsel 11: Ablation study bar chart
    - 4 config × ROC-AUC
    - Dosya: figures/11_ablation_study.png

Görsel 12: Fatigue trajectory örneği
    - Tek bir katılımcının 50 tur boyunca enerji + label değişimi
    - Dosya: figures/12_fatigue_trajectory.png
```

### 11.2 Tablolar (Makalede Kullanılacak)

```
Tablo 1: Dataset özeti
Tablo 2: Feature listesi + dağılım bilgisi
Tablo 3: Model hyperparameter sonuçları
Tablo 4: Ana sonuç tablosu (5 model × 6 metrik)
Tablo 5: Ablation study sonuçları
Tablo 6: McNemar test sonuçları (10 çift)
Tablo 7: SHAP top-10 feature sıralaması
```

---

## 12. Discussion & Conclusion

### 12.1 Beklenen Sonuç Yorumları

**XGBoost neden en iyi olacak:**  
Gradient boosting, sequential learning ile yorgunluğun kümülatif ve nonlinear doğasını yakalamada üstün. `scale_pos_weight` parametresi class imbalance'ı etkin yönetir. DDM feature'larındaki nonlinear etkileşimler (drift_boundary_ratio × session_position) boosting'de ağırlanır.

**Random Forest neden yakın olacak:**  
Ensemble yapısı gürültüye dirençli. Ama DDM feature'larının korelasyonu ağaç seçiminde redundancy yaratır ve XGBoost'u tam geçemez.

**Logistic Regression neden geride kalacak:**  
Tipping point nonlinearity (fatigue_level > 0.7 → ani artış) ve etkileşim terimleri (temporal_load_interaction) lineer modelle yakalanamaz. Bu beklenen ve kabul edilebilir bir bulgu — Discussion'da açıklanır.

**KNN neden en kötü olacak:**  
Temporal sinyaller (rt_slope, fatigue_slope) zaman boyutunda anlam taşır ama KNN'nin Euclidean mesafesi bu yapıyı görmez. Yüksek boyut (24 feature) curse of dimensionality etkisi yaratır.

**Ablation bulgusu:**  
Config B (No DDM) vs Config A (Full) arasındaki ROC-AUC farkı Δ₁ istatistiksel olarak anlamlı ise → RQ2 yanıtlanmıştır: DDM feature'ları ekstra tahmin gücü sağlar.

### 12.2 Limitations

```
1. Sentetik veri kısıtı:
   Gerçek insan davranışı daha karmaşık confound değişkenler içerir
   (motivasyon, uyku, gün içi dönem).

2. EZ-diffusion basitleştirmesi:
   Tam DDM (hDDM) hesaplaması değil, EZ-diffusion approximation kullanıldı.
   Daha kesin parametre tahmini için hDDM uygulanabilir.

3. Class imbalance:
   SMOTE sentetik azınlık örnekleri üretir — gerçek verinin dağılımını
   tam temsil etmeyebilir.

4. Single-session design:
   Kişiler arası öğrenme ve adaptasyon (inter-session) modellenmedi.

5. SVM subsampling:
   150K satır için SVM subsample ile çalıştırıldı.
   Tam veri üzerinde sonuçlar farklı olabilir.
```

### 12.3 Future Work

```
1. Gerçek kullanıcı verisi ile validasyon:
   Web uygulaması üzerinden mouse tıklama süreleri ve tercih örüntüleri
   toplanarak BDFS framework test edilebilir (IRB onayı ile).

2. Online / streaming detection:
   Gerçek zamanlı fatigue monitoring için sliding window + online learning.

3. Kişiselleştirilmiş eşik öğrenimi:
   individual_threshold parametresini gerçek veriden öğrenen adaptif sistem.

4. Multi-task learning:
   Hem fatigue sınıfını hem de cognitive energy skoru (regresyon) eş zamanlı tahmin.

5. Transformer-based sequential model:
   50 tur verisi time-series olarak işlendiğinde LSTM veya Transformer
   mevcut feature mühendisliği adımını atlayabilir.
```

---

## 13. Literatür Listesi (APA 7)

### Kategori 1: Bilişsel Yük ve Karar Yorgunluğu

Baumeister, R. F., Bratslavsky, E., Muraven, M., & Tice, D. M. (1998). Ego depletion: Is the active self a limited resource? *Journal of Personality and Social Psychology, 74*(5), 1252–1265. https://doi.org/10.1037/0022-3514.74.5.1252

Hagger, M. S., Chatzisarantis, N. L. D., Alberts, H., Anggono, C. O., Batailler, C., Birt, A., Brand, R., Brandt, M. J., Brewer, G., & Bruyneel, S. (2016). A multilab preregistered replication of the ego-depletion effect. *Perspectives on Psychological Science, 11*(4), 546–573. https://doi.org/10.1177/1745691616652873

Persson, E., Barrafrem, K., Meunier, A., & Tinghög, G. (2019). The effect of decision fatigue on surgeons' clinical decision making. *Health Economics, 28*(10), 1194–1203. https://doi.org/10.1002/hec.3933

Maier, M., Powell, D., Harrison, C., Gordon, J., Murchie, P., & Allan, J. L. (2024). Assessing decision fatigue in general practitioners' prescribing decisions using the Australian BEACH data set. *Medical Decision Making, 44*(6), 673–683. https://doi.org/10.1177/0272989X241263823

Plonsky, O., Apel, R., Ert, E., Tennenholtz, M., Bourgin, D., Peterson, J. C., Reichman, D., Griffiths, T. L., Russell, S. J., Carter, E. C., Cavanagh, J. F., Ert, E., & Erev, I. (2019). Predicting human decisions with behavioral theories and machine learning. *arXiv preprint arXiv:1904.06866*.

Iyengar, S. S., & Lepper, M. R. (2000). When choice is demotivating: Can one desire too much of a good thing? *Journal of Personality and Social Psychology, 79*(6), 995–1006. https://doi.org/10.1037/0022-3514.79.6.995

### Kategori 2: Drift Diffusion Model ve Bilişsel Modelleme

Ratcliff, R., & McKoon, G. (2008). The diffusion decision model: Theory and data for two-choice decision tasks. *Neural Computation, 20*(4), 873–922. https://doi.org/10.1162/neco.2008.12-06-420

Wagenmakers, E.-J., van der Maas, H. L. J., & Grasman, R. P. P. P. (2007). An EZ-diffusion model for response time and accuracy. *Psychonomic Bulletin & Review, 14*(1), 3–22. https://doi.org/10.3758/BF03194023

Vandekerckhove, J., Tuerlinckx, F., & Lee, M. D. (2011). Hierarchical diffusion models for two-choice response times. *Psychological Methods, 16*(1), 44–62. https://doi.org/10.1037/a0021765

Forstmann, B. U., Ratcliff, R., & Wagenmakers, E.-J. (2016). Sequential sampling models in cognitive neuroscience: Advantages, applications, and extensions. *Annual Review of Psychology, 67*, 641–666. https://doi.org/10.1146/annurev-psych-122414-033645

Matzke, D., & Wagenmakers, E.-J. (2009). Psychological interpretation of the ex-Gaussian and shifted Wald parameters: A diffusion model analysis. *Psychonomic Bulletin & Review, 16*(5), 798–817. https://doi.org/10.3758/PBR.16.5.798

### Kategori 3: ML Algoritmaları

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297. https://doi.org/10.1007/BF00994018

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*, 4765–4774.

Lundberg, S. M., Erion, G. G., & Lee, S.-I. (2018). Consistent individualized feature attribution for tree ensembles. *arXiv preprint arXiv:1802.03888*.

### Kategori 4: Sentetik Veri ve Evaluation

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357. https://doi.org/10.1613/jair.953

Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *Proceedings of the International Joint Conference on Artificial Intelligence, 14*, 1137–1145.

McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika, 12*(2), 153–157. https://doi.org/10.1007/BF02295996

### Kategori 5: Bilişsel Yük — Sensörlü ML Çalışmaları (Karşılaştırma Grubu)

Chakladar, D. D., Dey, S., Roy, P. P., & Dogra, D. P. (2022). A multimodal deep clustering framework for cognitive load estimation. *IEEE Transactions on Cognitive and Developmental Systems, 14*(3), 1151–1163. https://doi.org/10.1109/TCDS.2021.3078082

Oppelt, M. P., Foltyn, A., Deuschel, J., Lang, N. R., Holzer, N., Eskofier, B. M., & Yang, S. H. (2023). ADABase: A multimodal dataset for cognitive load estimation. *Sensors, 23*(1), 340. https://doi.org/10.3390/s23010340

Demirezen, G., Taşkaya Temizel, T., & Brouwer, A.-M. (2024). Reproducible machine learning research in mental workload classification using EEG. *Frontiers in Neuroergonomics, 5*, 1346794. https://doi.org/10.3389/fnrgo.2024.1346794

Yoo, G., Kim, H., & Hong, S. (2023). Prediction of cognitive load from electroencephalography signals using long short-term memory network. *Bioengineering, 10*(3), 361. https://doi.org/10.3390/bioengineering10030361

---

## 14. Dosya Yapısı

```
BDFS_Project/
│
├── config.py                   ← Tüm sabitler
├── README.md                   ← Proje açıklaması
│
├── data/
│   ├── raw/
│   │   └── bdfs_raw.csv        ← Üretilen ham veri
│   ├── processed/
│   │   ├── bdfs_train.csv
│   │   ├── bdfs_val.csv
│   │   └── bdfs_test.csv
│   └── splits/
│       └── split_indices.pkl   ← Reproducibility için kayıtlı split indeksleri
│
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_eda_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   ├── 06_ablation_study.ipynb
│   └── 07_shap_interpretability.ipynb
│
├── src/
│   ├── data_generator.py       ← Veri üretim fonksiyonları
│   ├── preprocessor.py         ← Preprocessing pipeline
│   ├── feature_engineer.py     ← Feature engineering
│   ├── models.py               ← 5 model + pipeline tanımları
│   ├── evaluator.py            ← Metrics + görsel üretimi
│   └── ablation.py             ← Ablation study runner
│
├── figures/                    ← Tüm görseller (1-12)
│   └── *.png
│
├── results/
│   ├── model_comparison.csv    ← Ana sonuç tablosu
│   ├── ablation_results.csv    ← Ablation tablosu
│   ├── mcnemar_results.csv     ← İstatistiksel test sonuçları
│   └── shap_values.pkl         ← SHAP değerleri (kaydedilmiş)
│
├── models/
│   ├── lr_best.pkl
│   ├── rf_best.pkl
│   ├── xgb_best.pkl
│   ├── svm_best.pkl
│   └── knn_best.pkl
│
└── paper/
    ├── BDFS_manuscript.docx    ← Makale taslağı
    └── references.bib          ← Zotero uyumlu BibTeX
```

---

## 15. Claude'a Adım Adım Prompt Sırası

Projeyi Claude'a bu sırayla ver. Her adımı tamamla, çıktıyı kaydet, sonraki adıma geç.

### Adım 1 — Veri Üretimi

```
Prompt:
"Python ile BDFS projesi için sentetik veri üret.
- N_PARTICIPANTS = 3000, N_TRIALS = 50
- config.py dosyasındaki tüm sabitleri kullan
- Ornstein-Uhlenbeck süreci ile cognitive_energy güncelle
- 18 feature üret: rt_mean, rt_std, rt_slope, rt_skew, rt_kurtosis,
  pref_reversal_rate, rolling_incon_5, rolling_incon_10, choice_entropy,
  session_position, fatigue_slope, accuracy_decay_rate,
  inter_trial_variability, ez_drift_rate, ez_boundary, ez_nondecision,
  drift_boundary_ratio, task_complexity_level
- EZ-diffusion formüllerini uygula
- %4 missing value, %3 outlier ekle
- Çıktı: bdfs_raw.csv
- random_state=42 her yerde
- Kod çalıştırılabilir olsun, yorum satırları Türkçe"
```

### Adım 2 — EDA ve Görselleştirme

```
Prompt:
"bdfs_raw.csv dosyasını yükle ve şunları yap:
- Her feature'ın dağılımını çiz (histogram, 0/1 sınıf ayrı renkle)
- Korelasyon matrisi oluştur (seaborn heatmap)
- Class distribution bar chart
- Eksik değer ve outlier raporu (her feature için yüzde)
- Tüm görselleri figures/ klasörüne kaydet
- Summary istatistikleri pandas describe() ile yazdır"
```

### Adım 3 — Preprocessing

```
Prompt:
"bdfs_raw.csv üzerinde preprocessing yap:
- Median imputation (eksik değerler)
- IQR capping (outlier — silme, winsorize et)
- StandardScaler (sadece train'e fit et)
- task_complexity_level → one-hot encoding (drop='first')
- Stratified train/val/test split: %70/%15/%15
- SMOTE sadece train setine uygula
- Split indekslerini split_indices.pkl olarak kaydet
- X_train, X_val, X_test, y_train, y_val, y_test oluştur
- Her setin class dağılımını yazdır (kontrol)"
```

### Adım 4 — Feature Engineering

```
Prompt:
"Preprocessing sonrası feature engineering yap:
- Pearson korelasyon matrisi + yüksek korelasyon raporu (|r|>0.8)
- VIF hesapla, VIF>10 olanları listele
- Mutual information skorları hesapla + görselleştir
- 6 yeni feature türet:
  rt_cv, incon_acceleration, ddm_fatigue_index,
  temporal_load_interaction, rt_kurtosis_normalized, decision_efficiency
- RFE ile top-15 feature seç (RF ile)
- Feature listesi ve sıralamasını yazdır"
```

### Adım 5 — Model Eğitimi

```
Prompt:
"5 ML modeli eğit. Her biri için sklearn Pipeline kullan.
Model 1: LogisticRegression, GridSearchCV
Model 2: RandomForestClassifier, RandomizedSearchCV n_iter=50
Model 3: XGBoostClassifier, RandomizedSearchCV n_iter=60
Model 4: SVC(probability=True), GridSearchCV, X_train'den 30000 subsample
Model 5: KNeighborsClassifier, GridSearchCV
- n_jobs=-1 paralel işlem
- random_state=42
- Her modelin en iyi parametrelerini yazdır
- Best modelleri models/ klasörüne joblib ile kaydet"
```

### Adım 6 — Evaluation

```
Prompt:
"5 best model için evaluation yap:
- StratifiedKFold (k=10) ile CV: accuracy, precision, recall, f1, roc_auc
  Her metrik için mean±std yazdır
- X_test üzerinde hold-out evaluation
- 5 model tek grafikte ROC curve
- 5 model için confusion matrix (1×5 subplot)
- Model karşılaştırma tablosu (DataFrame olarak yazdır + CSV kaydet)
- McNemar testi: tüm model çiftleri (Bonferroni düzeltmeli)
- Tüm görselleri figures/ klasörüne kaydet"
```

### Adım 7 — Ablation Study

```
Prompt:
"Ablation study yap. 4 feature config ile XGBoost ve RF'yi tekrar eğit:
Config A: Tüm 24 feature
Config B: DDM feature'ları çıkarılmış (20 feature)
Config C: Temporal feature'lar çıkarılmış (20 feature)
Config D: Sadece 5 baseline feature
- Her config için ROC-AUC ve F1 hesapla
- Sonuçları ablation_results.csv olarak kaydet
- Grouped bar chart ile görselleştir (figures/11_ablation_study.png)"
```

### Adım 8 — SHAP

```
Prompt:
"XGBoost best model için SHAP analizi yap:
- shap.TreeExplainer kullan
- Summary plot (beeswarm) → figures/09_shap_summary.png
- Bar plot (ortalama mutlak SHAP) → kaydet
- ez_drift_rate için dependence plot → figures/10_shap_dependence.png
- İlk fatigued örnek için waterfall plot
- SHAP top-10 feature sıralamasını DataFrame olarak yazdır
- RF feature importance ile Spearman rank korelasyonu hesapla"
```

### Adım 9 — Makale Tabloları

```
Prompt:
"Makale için şu tabloları hazırla (Markdown formatında):
Tablo 1: Dataset özeti (N, feature sayısı, class dağılımı)
Tablo 2: 18 ham feature listesi + dağılım bilgisi
Tablo 3: Model hyperparameter seçimleri (GridSearch sonucu)
Tablo 4: Ana sonuç tablosu (5 model × 6 metrik, en iyi kalın)
Tablo 5: Ablation study sonuçları
Tablo 6: McNemar testi p-değerleri matrisi
Tablo 7: SHAP top-10 feature sıralaması"
```

---

*Hazırlayan: BDFS Research Team*  
*Framework: CRISP-DM | Dil: Python 3.11 | Reproducibility: seed(42)*  
*Hedef dergi: Frontiers in Cognitive Science / IEEE Access / PLOS ONE*
