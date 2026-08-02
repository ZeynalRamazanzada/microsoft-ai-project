# config.py
# BDFS Projesi — Tüm sabitler tek dosyada toplanır
# Pipeline boyunca bu dosyadan çağrılır

# ==================== TEMEL PARAMETRELER ====================
RANDOM_SEED = 42                              # Reproducibility için sabit seed
N_PARTICIPANTS = 3000                         # Sentetik katılımcı sayısı
N_TRIALS = 50                                 # Her katılımcı için tur sayısı
N_TOTAL = N_PARTICIPANTS * N_TRIALS           # Toplam kayıt: 150.000

# ==================== SINIF DAĞILIMI ====================
CLASS_RATIO = 0.35                            # Pozitif sınıf oranı (fatigued)

# ==================== VERİ BÖLÜMLEMESİ ====================
TEST_SIZE = 0.15                              # Test seti oranı
VAL_SIZE = 0.15                               # Validation seti oranı
CV_FOLDS = 10                                 # Cross-validation katlama sayısı

# ==================== GÜRÜLTÜ VE AYKIRI DEĞER ====================
NOISE_LEVEL = 0.08                            # Genel gürültü seviyesi
OUTLIER_FRACTION = 0.03                       # Aykırı değer oranı (%3)
MISSING_FRACTION = 0.04                       # Eksik değer oranı (%4)

# ==================== YORGUNLUK EŞİĞİ ====================
FATIGUE_THRESHOLD_MULTIPLIER = 0.11           # Label üretiminde eşik çarpanı (kalibre edilmiş)

# ==================== ORNSTEIN-UHLENBECK SÜRECİ ====================
OU_THETA = 0.3                                # Mean reversion hızı
OU_MU = 0.2                                   # Uzun vadeli ortalama
OU_SIGMA = 0.05                               # Volatilite
OU_DT = 1 / N_TRIALS                          # Zaman adımı (1/50)
BASE_DECAY = 0.004                            # Temel enerji azalma oranı (kalibre edilmiş)
DECAY_NOISE_STD = 0.02                        # Enerji azalma gürültüsü

# ==================== DDM PARAMETRELERİ ====================
DDM_DRIFT_CLIP = (0.05, 1.0)                  # Drift rate kırpma aralığı
DDM_BOUNDARY_CLIP = (0.5, 2.0)               # Boundary kırpma aralığı
DDM_NONDECISION_CLIP = (0.05, 0.5)           # Non-decision time kırpma aralığı

# ==================== SESSION LOAD DAĞILIMI ====================
SESSION_LOAD_PROBS = [0.20, 0.50, 0.30]       # Düşük/Orta/Yüksek yük olasılıkları

# ==================== DOSYA YOLLARI ====================
RAW_DATA_PATH = "data/raw/bdfs_raw.csv"
TRAIN_DATA_PATH = "data/processed/bdfs_train.csv"
VAL_DATA_PATH = "data/processed/bdfs_val.csv"
TEST_DATA_PATH = "data/processed/bdfs_test.csv"
SPLIT_INDICES_PATH = "data/splits/split_indices.pkl"
FIGURES_DIR = "figures/"
RESULTS_DIR = "results/"
MODELS_DIR = "models/"
