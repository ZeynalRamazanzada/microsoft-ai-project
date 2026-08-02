# Kapsamlı Literatür Taraması
## Karar Yorgunluğu, DDM ve Bilişsel Yük için Makine Öğrenmesi Üzerine Sistematik Literatür Taraması
### Davranışsal Karar Yorgunluğu Skorlaması (BDFS) İçin Sistematik Bir İnceleme

**Proje:** BDFS — Sentetik Sıralı Seçim Verileri Üzerinde Sensörsüz Çoklu Sinyal Makine Öğrenmesi Kullanarak Davranışsal Karar Yorgunluğu Skorlama Çerçevesi  
**Hazırlanan Düzey:** Akademik Değerlendirme (Yüksek Lisans Seviyesi)  
**Tarih:** Haziran 2026  
**Ekip:** BDFS Araştırma Ekibi  

---

## İçindekiler

1. [Faz 1 — Proje Analizi](#faz-1--proje-analizi)
2. [Faz 2 — Literatür Tarama Stratejisi](#faz-2--literatür-tarama-stratejisi)
3. [Faz 3 — Eleştirel Literatür Analizi](#faz-3--eleştirel-literatür-analizi)
4. [Faz 4/5 — Detaylı Kaynak Kartları](#faz-45--detaylı-kaynak-kartları)
5. [Faz 6 — Nihai Çıktılar](#faz-6--nihai-çıktılar)
   - 6.1 [Profesyonel Düzeyde Yazılmış Literatür Taraması Özeti](#61-profesyonel-düzeyde-yazılmış-literatür-taraması-özeti)
   - 6.2 [Tam Referanslar — APA 7](#62-tam-referanslar--apa-7)
   - 6.3 [Kategorize Edilmiş Literatür Özeti](#63-kategorize-edilmiş-literatür-özeti)
   - 6.4 [Belirlenen Araştırma Boşlukları](#64-belirlenen-araştırma-boşlukları)
   - 6.5 [Özgün Katkı Önerileri](#65-özgün-katkı-önerileri)
   - 6.6 [Gelecek Araştırmalar İçin Öneriler](#66-gelecek-araştırmalar-için-öneriler)

---

# Faz 1 — Proje Analizi

## 1.1 Temel Araştırma Problemi

Bu araştırmanın temel problemi, bilişsel bilim ve makine öğrenmesinin kesişim noktasındaki kritik bir boşluğu ele almaktadır: **Sıralı karar verme süreçlerindeki bilişsel yorgunluk durumları, fizyolojik sensörler kullanılmadan, yalnızca davranışsal sinyallerden ve hesapsal olarak türetilmiş gizli bilişsel özelliklerle zenginleştirilmiş klasik makine öğrenmesi modelleri ile güvenilir bir şekilde tahmin edilebilir mi?**

Karar yorgunluğu psikolojide iyi belgelenmiş bir olgudur. Ancak mevcut tespit yöntemleri iki izole paradigmaya bölünmüştür: (1) EEG, fNIRS veya ECG ile derin öğrenme kullanan sensör bağımlı yaklaşımlar ve (2) hesapsal modellemeden yoksun olan ve sosyal bilimlerde yürütülen gözlemsel davranış çalışmalar. BDFS projesi, açıkça bu doldurulmamış metodolojik alanı hedeflemektedir.

## 1.2 Akademik Disiplinler

| Disiplin | İlgililik Durumu |
|:---|:---|
| **Bilişsel Psikoloji** | Karar yorgunluğu teorisi, ego tükenmesi (ego depletion), seçenek aşırı yüklenmesi |
| **Matematiksel Psikoloji** | Sürüklenme-Difüzyon Modeli (DDM), kanıt birikimi |
| **Makine Öğrenmesi / Veri Bilimi** | Sınıflandırma algoritmaları, özellik mühendisliği, model değerlendirmesi |
| **Açıklanabilir Yapay Zeka (XAI)** | SHAP değerleri, model yorumlanabilirliği |
| **Davranış Bilimleri** | Gözlemlenebilir davranışsal belirteçler, tepki süresi analizi |
| **Hesapsal Bilişsel Bilim** | EZ-difüzyon parametre tahmini, sentetik veri üretimi |

## 1.3 İlgili Teoriler, Teknolojiler ve Çerçeveler

### Teorik Çerçeveler
- **Ego Tükenmesi / Özdenetim Güç Modeli** (Baumeister vd., 1998)
- **Tükenmenin Süreç Modeli** (Inzlicht & Schmeichel, 2012)
- **Seçenek Aşırı Yüklenmesi Hipotezi** (Iyengar & Lepper, 2000)
- **Sıralı Örnekleme / Kanıt Birikimi** (Ratcliff & McKoon, 2008)

### Hesapsal Çerçeveler
- **Sürüklenme-Difüzyon Modeli (DDM)**: Gizli parametreler (sürüklenme hızı *v*, sınır ayrımı *a*, karar dışı süre *T*_er)
- **EZ-Difüzyon Modeli** (Wagenmakers vd., 2007)
- **Ornstein-Uhlenbeck Süreci**: Stokastik ortalamaya dönen süreç

### Makine Öğrenmesi Metodolojileri
- **Topluluk (Ensemble) Yöntemleri**: Random Forest, XGBoost
- **SMOTE**: Sentetik Azınlık Aşırı Örnekleme Tekniği
- **SHAP**: SHapley Additive exPlanations

## 1.4 Potansiyel Araştırma Soruları

1. Sıralı karar verme sırasındaki bilişsel yorgunluk, fizyolojik sensörler olmadan gözlemlenebilir davranışsal kalıplardan ne ölçüde tahmin edilebilir?
2. EZ-Difüzyon Modelinden türetilen gizli özellikler, yalnızca ham davranışsal özelliklere kıyasla anlamlı bir ek tahmin gücü sağlar mı?

## 1.5 Yenilikçi/Özgün Yönler

- **Sensörsüz bilişsel durum tespiti**: Mevcut bilişsel yük izleme sistemlerini sınırlayan donanım bariyerini ortadan kaldırır.
- **Makine öğrenmesi girdisi olarak DDM türevli gizli özellikler**: Hesapsal bilişsel bilim ile uygulamalı makine öğrenmesini birbirine bağlar.
- **Sistematik ablasyon (eksiltme) tasarımı**: DDM özelliklerinin, zamansal özelliklerin ve davranışsal temel özelliklerin marjinal katkısını nicel olarak ölçer.
- **Temel doğrusu (ground truth) bilinen kontrollü sentetik veri**: Tespit yeteneklerinin hassas bir şekilde değerlendirilmesine olanak tanır.

## 1.6 Anahtar Kelimeler ve Arama Stratejisi

### Temel Anahtar Kelimeler
`karar yorgunluğu`, `ego tükenmesi`, `bilişsel yük tespiti`, `sürüklenme difüzyon modeli`, `EZ-difüzyon`, `zihinsel iş yükü sınıflandırması`, `sensörsüz yorgunluk tespiti`

### Alternatif Akademik Anahtar Kelimeler
`seçenek aşırı yüklenmesi`, `sıralı karar verme`, `bilişsel tükenme`, `zihinsel yorgunluk makine öğrenmesi`, `tepki süresi yorgunluğu`, `dijital biyobelirteç yorgunluk`, `açıklanabilir yapay zeka bilişsel`

### Mantıksal (Boolean) Arama Kombinasyonları

```
("decision fatigue" OR "ego depletion") AND ("machine learning" OR "classification")

("drift diffusion model" OR "EZ-diffusion") AND ("fatigue" OR "cognitive load" OR "sleep deprivation")

("cognitive load" OR "mental workload") AND ("detection" OR "classification") AND ("sensor-free" OR "behavioral" OR "without sensors")

("reaction time" OR "response time") AND ("fatigue" OR "cognitive decline") AND ("prediction" OR "machine learning")
```

---

# Faz 2 — Literatür Tarama Stratejisi

## 2.1 Veritabanı Arama Planı

| Veritabanı | Odak Alanları | Beklenen Verim |
|:---|:---|:---|
| **Google Scholar** | Geniş disiplinlerarası arama; atıf doğrulama | Yüksek — birincil keşif aracı |
| **IEEE Xplore** | Sensör tabanlı ML, sinyal işleme | Yüksek — mühendislik literatürü |
| **ACM Digital Library** | HCI, davranışsal sinyaller, CHI bildiri kitapları | Orta-Yüksek |
| **PubMed / PsycINFO** | Psikoloji, nörobilim, bilişsel modelleme | Yüksek — teorik temel |
| **Springer** | Makine Öğrenmesi dergileri, bilişsel bilimler | Orta |
| **ScienceDirect (Elsevier)** | Neuroscience & Biobehavioral Reviews | Yüksek |
| **Nature** | Nature Machine Intelligence (SHAP) | Orta |

## 2.2 Dahil Etme Kriterleri

- Hakemli dergi makaleleri veya en üst düzey konferans bildirileri.
- Q1/Q2 dergilerinde (Scopus/SJR aracılığıyla doğrulanmış) yayınlanmış olması.
- Yaşına göre yüksek atıf sayısına sahip olması (3 yıldan eski makaleler için en az ~50 atıf).
- Karar yorgunluğu, DDM, bilişsel yük için makine öğrenmesi veya makine öğrenmesi metodolojisi (SMOTE/SHAP) ile doğrudan ilişkili olması.

## 2.3 Hariç Tutma Kriterleri

- Hakemli olmayan kaynaklar (bloglar, teknik incelemeler, fikir yazıları).
- Yetersiz metodolojik titizliğe sahip çalışmalar.

---

# Faz 3 — Eleştirel Literatür Analizi

## 3.1 Tematik Analiz: Karar Yorgunluğu ve Ego Tükenmesi

Karar yorgunluğu kavramı, özdenetimin sonlu, tükenebilir bir kaynağa dayandığını öne süren Baumeister vd.'nin (1998) özdenetim güç modeline dayanmaktadır. Vohs vd. (2008) karar vermenin kendisinin de tüketici olduğunu göstererek bunu genişletmiştir. Bu durumun gerçek dünyadaki uygulanabilirliği, Danziger vd. (2011) tarafından adli şartlı tahliye kararları analizinde çarpıcı bir şekilde gösterilmiştir.

**Çelişkili bulgular.** Teorik temeller, ego tükenmesi etkisi için kanıt bulamayan ön kayıtlı (preregistered) çok merkezli bir replikasyon çalışması olan Hagger vd. (2016) tarafından önemli ölçüde sorgulanmıştır. Buna yanıt olarak Inzlicht ve Schmeichel (2012), tükenmenin kaynak tükenmesinden ziyade motivasyon ve dikkatteki kaymaları yansıttığını savunan alternatif bir süreç modeli önermişlerdir.

**BDFS İçin Doğurguları.** BDFS, altta yatan mekanizmaya agnostik (tarafsız) olacak şekilde konumlandırılmıştır — yorgunluğun nedeni ister kaynak tükenmesi ister motivasyonel kayma olsun, yorgunlukla ilişkili davranışsal kalıpları tespit eder.

## 3.2 Tematik Analiz: Sürüklenme-Difüzyon Modelleri (DDM)

Sürüklenme-Difüzyon Modeli (DDM), BDFS'yi tamamen ampirik davranışsal yaklaşımlardan ayıran hesapsal bilişsel çerçeveyi sağlar. Ratcliff ve McKoon (2008), iki seçenekli görevler için DDM'nin kesin formülizasyonunu sağlamıştır. Wagenmakers vd. (2007), özet istatistiklerden DDM parametrelerini tahmin etmek için basitleştirilmiş bir yaklaşım olan EZ-Difüzyon Modelini sunarak bunu ML özellik mühendisliği için pratik hale getirmiştir.

DDM parametreleri ile yorgunluk arasındaki doğrudan bağlantı, yorgunluğun sürüklenme oranını azalttığını ve karar dışı süre ile değişkenliği artırdığını gösteren Ratcliff ve Van Dongen (2009, 2011) tarafından kurulmuştur. Bu, DDM parametrelerinin yorgunluğun farklı boyutlarını yakaladığını ampirik olarak doğrulamaktadır.

## 3.3 Tematik Analiz: Bilişsel Yük Tespiti İçin Makine Öğrenmesi

Bilişsel yük tespitinde baskın yaklaşım fizyolojik sinyallere dayanmaktadır (Borghini vd., 2014; Dolmans vd., 2021). Ancak, yeni ortaya çıkan sensörsüz yaklaşımlar, bilişsel durumların yalnızca davranışsal sinyallerden tespit edilmesinin uygulanabilirliğini göstermektedir. Fridman vd. (2018) kamera kaynaklı özelliklerden makul bir sınıflandırma elde etmiş ve Acien vd. (2022) tuş vuruşu dinamiklerinin zihinsel yorgunluk için dijital bir biyobelirteç olabileceğini kanıtlamıştır.

**Araştırma Boşluğu.** Mevcut hiçbir çalışma, DDM türevli gizli özellikleri sensörsüz bilişsel yorgunluk tespiti için klasik makine öğrenmesi modelleriyle birleştirmemektedir. BDFS, bu yaklaşımlar arasında benzersiz bir köprü kurmaktadır.

## 3.4 Tematik Analiz: Makine Öğrenmesi Metodolojisi

BDFS'deki makine öğrenmesi algoritma seçimleri, özellikle Random Forest (Breiman, 2001) ve XGBoost (Chen & Guestrin, 2016) olmak üzere sağlam temellere dayanmaktadır. Sınıf dengesizliği SMOTE (Chawla vd., 2002) aracılığıyla ele alınmaktadır. Model yorumlanabilirliği, kesin ve teorik temelli özellik önemi sağlayan SHAP (Lundberg & Lee, 2017) ve TreeSHAP (Lundberg vd., 2020) aracılığıyla sağlanmaktadır.

---

# Faz 4/5 — Detaylı Kaynak Kartları

> [!NOTE]
> Aşağıdaki kaynak kartları, bu literatür taramasında kullanılan en önemli referanslar için ayrıntılı bilgi sağlamaktadır. Her kart başlık, yazarlar, yayın ayrıntıları, metodoloji, bulgular, güçlü yönler, sınırlamalar, projeyle ilişki ve atıf bilgilerini içerir.

---

### Kaynak Kartı 1: Baumeister vd. (1998)

| Alan | Detay |
|:---|:---|
| **Başlık** | Ego depletion: Is the active self a limited resource? |
| **Yazarlar** | Roy F. Baumeister, Ellen Bratslavsky, Mark Muraven, Dianne M. Tice |
| **Yıl** | 1998 |
| **Yayın** | *Journal of Personality and Social Psychology*, 74(5), 1252–1265 |
| **Atıf Sayısı** | ~9.200+ |
| **Araştırma Amacı** | Özdenetimin sınırlı, tükenebilir bir kaynağa dayanıp dayanmadığını test etmek |
| **Metodoloji** | Sıralı görev paradigmasını kullanan bir dizi laboratuvar deneyi |
| **Temel Bulgular** | Özdenetim sonlu bir kaynağı tüketir; sonraki görev performansı bozulur |
| **Güçlü Yönler** | Temel oluşturan çalışma; oldukça etkili |
| **Sınırlamalar** | Son dönemdeki replikasyon (tekrarlama) başarısızlıkları tarafından sorgulanmıştır |
| **Projeyle İlişkisi** | Karar vermenin bilişsel kaynakları tükettiğine dair temel teorik dayanağı sağlar |
| **DOI** | https://doi.org/10.1037/0022-3514.74.5.1252 |
| **APA 7 Atıf** | Baumeister, R. F., Bratslavsky, E., Muraven, M., & Tice, D. M. (1998). Ego depletion: Is the active self a limited resource? *Journal of Personality and Social Psychology*, *74*(5), 1252–1265. https://doi.org/10.1037/0022-3514.74.5.1252 |

---

### Kaynak Kartı 2: Vohs vd. (2008)

| Alan | Detay |
|:---|:---|
| **Başlık** | Making choices impairs subsequent self-control: A limited-resource account of decision making, self-regulation, and active initiative |
| **Yazarlar** | Kathleen D. Vohs, Roy F. Baumeister, Brandon J. Schmeichel, Jean M. Twenge, Noelle M. Nelson, Dianne M. Tice |
| **Yıl** | 2008 |
| **Yayın** | *Journal of Personality and Social Psychology*, 94(5), 883–898 |
| **Atıf Sayısı** | ~1.900+ |
| **Araştırma Amacı** | Karar verme eyleminin özdenetim kaynaklarını tükettiğini göstermek |
| **Metodoloji** | Seçim yapma görevlerinden sonra özdenetimi ölçen çoklu deneyler |
| **Temel Bulgular** | Seçim yapma eylemi, özdenetim için kullanılan aynı kaynağı tüketir |
| **Güçlü Yönler** | Karar vermeyi doğrudan tükenmeye (depletion) bağlar |
| **Sınırlamalar** | Laboratuvar paradigmasına dayanır |
| **Projeyle İlişkisi** | Sıralı karar sayısının yorgunluk skorlaması için birincil girdi olduğunu doğrular |
| **DOI** | https://doi.org/10.1037/0022-3514.94.5.883 |
| **APA 7 Atıf** | Vohs, K. D., Baumeister, R. F., Schmeichel, B. J., Twenge, J. M., Nelson, N. M., & Tice, D. M. (2008). Making choices impairs subsequent self-control: A limited-resource account of decision making, self-regulation, and active initiative. *Journal of Personality and Social Psychology*, *94*(5), 883–898. https://doi.org/10.1037/0022-3514.94.5.883 |

---

### Kaynak Kartı 3: Ratcliff & McKoon (2008)

| Alan | Detay |
|:---|:---|
| **Başlık** | The diffusion decision model: Theory and data for two-choice decision tasks |
| **Yazarlar** | Roger Ratcliff, Gail McKoon |
| **Yıl** | 2008 |
| **Yayın** | *Neural Computation*, 20(4), 873–922 |
| **Atıf Sayısı** | ~4.000+ |
| **Araştırma Amacı** | İki seçenekli tepki süresi görevleri için DDM'yi resmileştirmek |
| **Metodoloji** | Farklı bilişsel görevler arasında parametre tahmini ile matematiksel modelleme |
| **Temel Bulgular** | DDM tüm tepki süresi dağılımlarını, doğruluğu ve hız-doğruluk ödünleşimlerini açıklar |
| **Güçlü Yönler** | Kesin formülizasyon; kapsamlı gösterim |
| **Sınırlamalar** | Tam parametre tahmini hesaplama açısından pahalıdır |
| **Projeyle İlişkisi** | BDFS'nin makine öğrenmesi özellikleri olarak çıkardığı tüm DDM parametre uzayını tanımlar |
| **DOI** | https://doi.org/10.1162/neco.2008.12-06-420 |
| **APA 7 Atıf** | Ratcliff, R., & McKoon, G. (2008). The diffusion decision model: Theory and data for two-choice decision tasks. *Neural Computation*, *20*(4), 873–922. https://doi.org/10.1162/neco.2008.12-06-420 |

---

### Kaynak Kartı 4: Wagenmakers vd. (2007)

| Alan | Detay |
|:---|:---|
| **Başlık** | An EZ-diffusion model for response time and accuracy |
| **Yazarlar** | Eric-Jan Wagenmakers, Han L. J. van der Maas, Raoul P. P. P. Grasman |
| **Yıl** | 2007 |
| **Yayın** | *Psychonomic Bulletin & Review*, 14(1), 3–22 |
| **Atıf Sayısı** | ~2.500+ |
| **Araştırma Amacı** | Momentler yöntemi (method-of-moments) DDM tahmin yaklaşımı önermek |
| **Metodoloji** | Ortalama tepki süresi, tepki süresi varyansı ve doğruluktan v, a, T_er'yi tahmin eden kapalı form denklemler |
| **Temel Bulgular** | EZ-difüzyon, hızlı ve erişilebilir parametre tahmini sağlar |
| **Güçlü Yönler** | Hesapsal basitlik; yalnızca özet istatistikler gerektirir |
| **Sınırlamalar** | Tam DDM'den daha az hassastır |
| **Projeyle İlişkisi** | Doğrudan metodolojik temel: BDFS özellik çıkarımı için EZ-difüzyon kullanır |
| **DOI** | https://doi.org/10.3758/BF03194023 |
| **APA 7 Atıf** | Wagenmakers, E.-J., van der Maas, H. L. J., & Grasman, R. P. P. P. (2007). An EZ-diffusion model for response time and accuracy. *Psychonomic Bulletin & Review*, *14*(1), 3–22. https://doi.org/10.3758/BF03194023 |

---

### Kaynak Kartı 5: Acien vd. (2022)

| Alan | Detay |
|:---|:---|
| **Başlık** | Detection of mental fatigue in the general population: Feasibility study of keystroke dynamics as a real-world biomarker |
| **Yazarlar** | Alejandro Acien, Aythami Morales, Ruben Vera-Rodriguez, Julian Fierrez, Ijah Mondesire-Crump, Teresa Arroyo-Gallego |
| **Yıl** | 2022 |
| **Yayın** | *JMIR Biomedical Engineering* |
| **Atıf Sayısı** | Yeni artıyor |
| **Araştırma Amacı** | Zihinsel yorgunluk için müdahaleci olmayan dijital bir biyobelirteç olarak tuş vuruşu dinamiklerini göstermek |
| **Metodoloji** | SVM ve Random Forest ile tuş vuruşu özellik çıkarımı |
| **Temel Bulgular** | Günlük yazma kalıplarından yorgunluk tespiti için %70–90 doğruluk |
| **Güçlü Yönler** | Tamamen sensörsüz; ekolojik olarak geçerli |
| **Sınırlamalar** | Yalnızca tuş vuruşuna özgü |
| **Projeyle İlişkisi** | Davranışsal zamanlama sinyallerinden sensörsüz makine öğrenmesi yorgunluk tespitini doğrular |
| **DOI** | https://doi.org/10.2196/41003 |
| **APA 7 Atıf** | Acien, A., Morales, A., Vera-Rodriguez, R., Fierrez, J., Mondesire-Crump, I., & Arroyo-Gallego, T. (2022). Detection of mental fatigue in the general population: Feasibility study of keystroke dynamics as a real-world biomarker. *JMIR Biomedical Engineering*. https://doi.org/10.2196/41003 |

---

### Kaynak Kartı 6: Breiman (2001)

| Alan | Detay |
|:---|:---|
| **Başlık** | Random Forests |
| **Yazarlar** | Leo Breiman |
| **Yıl** | 2001 |
| **Yayın** | *Machine Learning*, 45(1), 5–32 |
| **Atıf Sayısı** | ~187.000+ |
| **Araştırma Amacı** | Random Forest topluluk (ensemble) öğrenme algoritmasını tanıtmak |
| **Metodoloji** | Rastgele özellik alt kümelerine sahip önyüklemeli (bootstrap-aggregated) karar ağaçları |
| **Temel Bulgular** | RF, gürültüye dayanıklılık ile rekabetçi doğruluk elde eder |
| **Güçlü Yönler** | Topluluk ML modelleri için temel oluşturur |
| **Sınırlamalar** | Değişken önemi yüksek kardinaliteli özelliklere karşı önyargılıdır |
| **Projeyle İlişkisi** | RF, BDFS boru hattındaki temel sınıflandırıcılardan biridir |
| **DOI** | https://doi.org/10.1023/A:1010933404324 |
| **APA 7 Atıf** | Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5–32. https://doi.org/10.1023/A:1010933404324 |

---

### Kaynak Kartı 7: Chawla vd. (2002)

| Alan | Detay |
|:---|:---|
| **Başlık** | SMOTE: Synthetic minority over-sampling technique |
| **Yazarlar** | Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, W. Philip Kegelmeyer |
| **Yıl** | 2002 |
| **Yayın** | *Journal of Artificial Intelligence Research*, 16, 321–357 |
| **Atıf Sayısı** | ~30.000+ |
| **Araştırma Amacı** | Sentetik azınlık aşırı örneklemesi ile sınıf dengesizliğini ele almak |
| **Metodoloji** | Azınlık örnekleri arasında özellik uzayı (feature-space) enterpolasyonu |
| **Temel Bulgular** | SMOTE azınlık sınıfı tespitini iyileştirir |
| **Güçlü Yönler** | Alanda temel oluşturan çalışma |
| **Sınırlamalar** | Gürültülü örnekler üretebilir |
| **Projeyle İlişkisi** | BDFS, eğitim seti dengelemesi için SMOTE kullanır |
| **DOI** | https://doi.org/10.1613/jair.953 |
| **APA 7 Atıf** | Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, *16*, 321–357. https://doi.org/10.1613/jair.953 |

---

### Kaynak Kartı 8: Lundberg & Lee (2017)

| Alan | Detay |
|:---|:---|
| **Başlık** | A unified approach to interpreting model predictions |
| **Yazarlar** | Scott M. Lundberg, Su-In Lee |
| **Yıl** | 2017 |
| **Yayın** | *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4768–4777 |
| **Atıf Sayısı** | ~55.000+ |
| **Araştırma Amacı** | Özellik atfetme (feature attribution) yöntemlerini oyun teorik Shapley değeri çerçevesinde birleştirmek |
| **Metodoloji** | Çeşitli yöntemleri birbirine bağlayan SHAP çerçevesi |
| **Temel Bulgular** | SHAP teorik temelli özellik önemi sağlar |
| **Güçlü Yönler** | Teorik zarafet; birleştirici çerçeve |
| **Sınırlamalar** | Optimizasyonlar olmadan büyük modeller için hesapsal olarak pahalıdır |
| **Projeyle İlişkisi** | BDFS modelleri için temel yorumlanabilirlik yöntemi |
| **DOI** | https://doi.org/10.48550/arXiv.1705.07874 |
| **APA 7 Atıf** | Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems* içinde (Cilt 30, ss. 4768–4777). |

---

# Faz 6 — Nihai Çıktılar

## 6.1 Profesyonel Düzeyde Yazılmış Literatür Taraması Özeti

*(Sentez Özeti)*
Sürüklenme-Difüzyon Modellerinin uygulamalı Makine Öğrenmesi ile entegrasyonu, bilişsel durum tespitinde yeni bir sınır (frontier) temsil etmektedir. Fizyolojik sensör tabanlı yaklaşımlar zihinsel iş yükü sınıflandırmasına hakim olsa da, ekolojik geçerlilikleri donanım kısıtlamaları nedeniyle sınırlıdır. BDFS, EZ-Difüzyon modeli aracılığıyla türetilen DDM parametreleriyle zenginleştirilmiş davranışsal sinyalleri — tepki süreleri ve seçim kalıpları — kullanarak bu sorunu aşar. Bu yaklaşım teorik olarak bilişsel psikolojiye dayanırken, pratik olarak Random Forest ve XGBoost gibi ölçeklenebilir ML sınıflandırma teknikleriyle uyumludur. SHAP tarafından sağlanan yorumlanabilirlik, ortaya çıkan modellerin kara kutu (black box) olmamasını, aksine yorgunluğun bilişsel belirteçlerini ortaya çıkarabilen teşhis araçları olmasını sağlar.

## 6.2 Tam Referanslar — APA 7

Acien, A., Morales, A., Vera-Rodriguez, R., Fierrez, J., Mondesire-Crump, I., & Arroyo-Gallego, T. (2022). Detection of mental fatigue in the general population: Feasibility study of keystroke dynamics as a real-world biomarker. *JMIR Biomedical Engineering*. https://doi.org/10.2196/41003

Baumeister, R. F., Bratslavsky, E., Muraven, M., & Tice, D. M. (1998). Ego depletion: Is the active self a limited resource? *Journal of Personality and Social Psychology*, *74*(5), 1252–1265. https://doi.org/10.1037/0022-3514.74.5.1252

Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, *16*, 321–357. https://doi.org/10.1613/jair.953

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* içinde (ss. 785–794). ACM.

Danziger, S., Levav, J., & Avnaim-Pesso, L. (2011). Extraneous factors in judicial decisions. *Proceedings of the National Academy of Sciences*, *108*(17), 6889–6892.

Hagger, M. S., vd. (2016). A multilab preregistered replication of the ego-depletion effect. *Perspectives on Psychological Science*, *11*(4), 546–573.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems* içinde (Cilt 30).

Ratcliff, R., & McKoon, G. (2008). The diffusion decision model: Theory and data for two-choice decision tasks. *Neural Computation*, *20*(4), 873–922.

Wagenmakers, E.-J., van der Maas, H. L. J., & Grasman, R. P. P. P. (2007). An EZ-diffusion model for response time and accuracy. *Psychonomic Bulletin & Review*, *14*(1), 3–22.

*(Not: 25 referansın tamamı oluşturulan RIS dosyasında sağlanmıştır)*

## 6.3 Kategorize Edilmiş Literatür Özeti

| Kategori | Önemli Makaleler | Temel İçgörü |
|:---|:---|:---|
| **Karar Yorgunluğu Teorisi** | Baumeister (1998), Vohs (2008) | Sıralı kararlar bilişsel kaynakları tüketir |
| **Sürüklenme-Difüzyon Modelleri** | Ratcliff & McKoon (2008), Wagenmakers (2007) | DDM parametreleri bilişsel durum göstergeleri olarak işlev görür |
| **Sensörsüz ML** | Acien vd. (2022), Fridman vd. (2018) | Davranışsal sinyaller yorgunluğu sensörsüz sınıflandırabilir |
| **ML Metodolojisi** | Breiman (2001), Chawla (2002), Lundberg (2017) | Ensemble + SMOTE + SHAP güçlü sınıflandırma boru hatları (pipelines) oluşturur |

## 6.4 Belirlenen Araştırma Boşlukları

> [!WARNING]
> ### Boşluk 1: DDM-ML Entegrasyonu Eksikliği
> Mevcut hiçbir çalışma, bilişsel yorgunluk tespiti için bir ML sınıflandırma boru hattı içinde DDM'den türetilmiş parametreleri tasarlanmış (engineered) özellikler olarak kullanmamaktadır.

> [!IMPORTANT]
> ### Boşluk 2: Yorgunluk ML Çalışmalarında Ablasyon Eksikliği
> Mevcut yorgunluk tespiti çalışmaları, bireysel özellik gruplarının katkısını nicel olarak belirlemek için nadiren sistematik ablasyon gerçekleştirir.

## 6.5 Özgün Katkı Önerileri

1. **İlk DDM-ML yorgunluk tespit çerçevesi**: Bilişsel bilim ve uygulamalı ML arasında köprü kurma.
2. **Sistematik ablasyon miktarının belirlenmesi**: DDM özellik değerleri için titiz kanıt.
3. **Teoriden bağımsız (theory-agnostic) tespit**: Kaynak ve süreç modeli tartışmasından bağımsız olarak yorgunluk kalıplarını yakalama.

## 6.6 Gelecek Araştırmalar İçin Öneriler

1. **Gerçek dünya doğrulaması**: BDFS'yi web tabanlı karar görevleri aracılığıyla toplanan gerçek insan davranış verilerinde devreye alma.
2. **Çevrimiçi/akan (streaming) tespit**: Kayan pencereler (sliding windows) kullanarak çerçeveyi gerçek zamanlı yorgunluk izlemeye genişletme.
3. **Derin öğrenme sıralı modeller**: Diziyi LSTM veya Transformer mimarileri kullanarak bir zaman serisi olarak işleme.

---
*Literatür Taramasının Sonu*
