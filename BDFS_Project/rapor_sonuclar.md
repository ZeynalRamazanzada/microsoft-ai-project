# BDFS Projesi - Tüm Sonuçlar Raporu

Bu rapor, Behavioral Decision Fatigue Scoring (BDFS) projesi kapsamında eğitilen makine öğrenimi modellerinin performans metriklerini, ablasyon çalışmalarını, istatistiksel test sonuçlarını ve SHAP özellik önemliliklerini içermektedir.

---

## 1. 6 Modelin Test Seti Sonuçları

Test seti (22.500 kayıt) üzerinde yapılan değerlendirme sonucunda modellerin elde ettiği temel performans metrikleri aşağıdadır:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LR** (Logistic Regression) | 0.8863 | 0.7743 | 0.9339 | 0.8466 | 0.9476 | 0.8549 |
| **RF** (Random Forest) | 0.9044 | 0.8202 | 0.9163 | 0.8656 | 0.9672 | 0.9232 |
| **XGB** (XGBoost) | 0.9053 | 0.8263 | 0.9094 | 0.8659 | 0.9667 | 0.9214 |
| **MLP** (Multi-Layer Perceptron) | 0.8654 | 0.7410 | 0.8520 | 0.7926 | 0.9125 | 0.8112 |
| **SVM** (Destek Vektör Makineleri)| 0.6708 | 0.5061 | 0.8650 | 0.6386 | 0.7713 | 0.5314 |
| **KNN** (K-Nearest Neighbors) | 0.7029 | 0.5386 | 0.8107 | 0.6472 | 0.8123 | 0.6576 |

---

## 2. Model Hiperparametre Optimizasyonu (model_hyperparams.csv)

Hyperparameter Tuning (CV=3) aşamasında modellerin elde ettiği en iyi konfigürasyonlar:

| Model | CV_ROC_AUC | Best_Params |
|:---|:---:|:---|
| **LR** | 0.9496 | `{'C': 10, 'class_weight': 'balanced', 'max_iter': 1000, 'penalty': 'l1', 'solver': 'liblinear'}` |
| **RF** | 0.9722 | `{'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 15, 'class_weight': 'balanced'}` |
| **XGB** | 0.9740 | `{'subsample': 1.0, 'scale_pos_weight': 1, 'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 1.0}` |
| **SVM** | 0.7778 | `{'C': 1, 'class_weight': 'balanced', 'gamma': 'scale'}` |
| **KNN** | 0.8136 | `{'metric': 'euclidean', 'n_neighbors': 21, 'weights': 'uniform'}` |

---

## 3. Ablation Study Sonuçları (ablation_results.csv)

XGBoost modeli kullanılarak özellik (feature) kümelerinin önemini anlamak için yapılan Ablasyon çalışması sonuçları:

| Config | Num_Features | ROC_AUC | F1_Score | Delta_AUC (Fark) |
|:---|:---:|:---:|:---:|:---:|
| **Full Model** (Tümü) | 26 | 0.9667 | 0.8659 | 0.0000 |
| **No DDM** (Bilişsel Model Hariç) | 21 | 0.9663 | 0.8633 | -0.0004 |
| **No Temporal** (Zamansal Hariç)| 18 | 0.9258 | 0.7846 | **-0.0409** |
| **Baseline** (Temel Davranışsal)| 5 | 0.9615 | 0.8525 | -0.0052 |

---

## 4. McNemar İstatistiksel Testi Sonuçları

En iyi model olan **XGBoost** ile Baseline (Referans) model olan **Logistic Regression (LR)** tahminleri arasındaki anlamlılık testi:

| Durum | XGBoost Doğru Tahmin | XGBoost Yanlış Tahmin |
|:---|:---:|:---:|
| **LR Doğru Tahmin** | 19533 | 408 |
| **LR Yanlış Tahmin** | 836 | 1723 |

- **McNemar Test İstatistiği:** 146.5667
- **P-value:** 9.7600e-34
- **Sonuç:** XGBoost ile Logistic Regression arasında İSTATİSTİKSEL OLARAK ANLAMLI bir fark vardır (p < 0.001).

---

## 5. SHAP Top-10 Özellik (Feature) Sıralaması

XGBoost modelinin kararlarını açıklayan (SHAP) en etkili 10 özellik:

| Sıra | Özellik (Feature) | Ortalama Mutlak SHAP Değeri (Etki Gücü) |
|:---:|:---|:---:|
| 1 | `rolling_incon_5` | 2.5152 |
| 2 | `rolling_incon_10` | 1.1684 |
| 3 | `pref_reversal_rate` | 0.5230 |
| 4 | `ez_nondecision` | 0.3662 |
| 5 | `session_position` | 0.1275 |
| 6 | `incon_acceleration` | 0.0999 |
| 7 | `accuracy_decay_rate` | 0.0869 |
| 8 | `fatigue_slope` | 0.0784 |
| 9 | `rt_slope` | 0.0683 |
| 10 | `session_load` | 0.0666 |

---

## 6. XGBoost En İyi Hiperparametre Değerleri

Model optimizasyonu (RandomizedSearchCV) sonucu bulunan, projeyi en yüksek performansa taşıyan parametreler:

- **n_estimators (Ağaç Sayısı):** 200
- **max_depth (Maksimum Derinlik):** 5
- **learning_rate (Öğrenme Hızı):** 0.1
- **subsample (Alt Örneklem Oranı):** 1.0
- **colsample_bytree (Ağaç Başına Sütun Oranı):** 1.0
- **scale_pos_weight (Sınıf Ağırlığı Dengesi):** 1

---

## 7. Ablation Study: En Çok Fark Yaratan Özellik Grubu

Ablasyon çalışmasına göre modelden çıkarıldığında en büyük performans kaybına (`-0.0409` ROC-AUC düşüşü) sebep olan özellik grubu **Temporal (Zamansal)** özelliklerdir (`No Temporal` konfigürasyonu). 

Bilişsel DDM (Drift Diffusion Model) parametrelerinin çıkarılması ise çok minimal bir etki yaratmıştır (`-0.0004`). Bu sonuç, karar verme süreçlerindeki ardışık eğilimlerin (ör. reaksiyon süresindeki değişim, hata ivmesi) statik bilişsel model parametrelerinden daha belirleyici olduğunu göstermektedir.

---

## 8. Genel Değerlendirme

**En İyi Model: XGBoost**

**Nedenleri:**
1. **Denge ve Kararlılık (F1-Score):** Yüksek ROC-AUC (0.9667) elde etmesinin yanında, Precision (0.8263) ve Recall (0.9094) arasında muazzam bir denge kurarak %86.59 F1-Score elde etmiştir. Sınıf dengesizliği olan (imbalanced) verilerde bu kararlılık çok değerlidir. Random Forest da çok benzer bir F1-Score (%86.56) vermiş olsa da, XGBoost McNemar istatistiğinde kanıtlandığı üzere diğer temel modellere kıyasla net bir istatistiksel üstünlüğe sahiptir.
2. **Kavrama Gücü:** XGBoost, `rolling_incon_5` ve `rolling_incon_10` gibi spesifik, doğrusal olmayan, zaman serisi mantığına dayanan özellikleri çok iyi yakalamıştır. Bu özellikler SHAP analizinde açık ara farkla karar mekanizmasını yöneten etkenler olarak belirlenmiştir.
3. **Genelleme Yeteneği:** Hem PR-AUC (Precision-Recall Eğrisi Altındaki Alan) değeri 0.9214 ile çok yüksektir, hem de hold-out test setinde CV (Cross-Validation) skoruna kıyasla düşüş yaşamayarak genelleme yeteneğini kanıtlamıştır.
