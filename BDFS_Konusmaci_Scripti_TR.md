# 🎤 BDFS — 10 Dakikalık Sunum Konuşmacı Scripti
### Behavioral Decision Fatigue Scoring (Davranışsal Karar Yorgunluğu Puanlama)
**Konuşmacılar:** Zeynal (Z) · Alperen (A)
**Toplam Süre:** ~10 dakika

---

## ⏱️ BÖLÜM 1 — ZAMANLAMA PLANI TABLOSU

| # | Slayt Başlığı | Konuşmacı | Süre | Kümülatif | Amaç |
|---|---|---|---|---|---|
| 1 | Başlık Slaytı | Zeynal | 0:35 | 0:35 | Açılış, takımı tanıtma |
| 2 | Görünmez Tehdit | Zeynal | 0:50 | 1:25 | Problemi ortaya koy |
| 3 | Neden Önemli | Zeynal | 0:45 | 2:10 | Riski büyüt |
| 4 | BDFS Çözümü | Zeynal | 0:40 | 2:50 | Çözümü tanıt, devret |
| 5 | Veri & Özellikler | Alperen | 0:55 | 3:45 | Teknik veri derinliği |
| 6 | Model Değerlendirme | Alperen | 1:00 | 4:45 | Sonuçlar, XGBoost neden kazandı |
| 7 | Sistem Pipeline | Alperen | 0:50 | 5:35 | Mimari anlatım |
| 8 | Backend Entegrasyonu | Alperen | 0:45 | 6:20 | FastAPI ve API tasarımı |
| 9 | CANLI DEMO | Alperen | 1:30 | 7:50 | Canlı tarayıcı demosu |
| 10 | Kullanıcı Arayüzü | Alperen | 0:40 | 8:30 | Demo sonrası UI anlatımı |
| 11 | Sonuçlar & SHAP | Zeynal | 0:45 | 9:15 | Temel bilimsel bulgu |
| 12 | Zorluklar & Çözümler | Zeynal | 0:30 | 9:45 | Güvenilirlik ve dürüstlük |
| 13 | Gelecek Planlar | Zeynal | 0:20 | 10:05 | Vizyon, hızlı geçiş |
| 14 | Sonuç & Sorular | Her İkisi | 0:25 | ~10:30 | Kapanış + teşekkür |

> **Not:** Canlı demo slaytı esnek tampon görevi görür. Süreniz azaldıysa demoyu uzatın, fazla gidiyorsa hazır profille kısa tutun.

---

## 🎙️ BÖLÜM 2 — TAM KONUŞMACI SCRIPTİ

---

### 🟦 SLAYT 1 — Başlık Slaytı
**Konuşmacı: ZEYNAL**
**Süre: ~35 saniye**

> *Slaytın 2 saniye ekranda kalmasını bekleyin, sonra konuşmaya başlayın.*

---

**Söylenecekler:**

"Günaydın herkese. Ben Zeynal, yanımdaki arkadaşım Alperen. Bugün sizlere BDFS'i — yani Behavioral Decision Fatigue Scoring'i, Türkçesiyle Davranışsal Karar Yorgunluğu Puanlama sistemini sunuyoruz.

Başlarken şu soruyu sormak istiyorum: Birinin zihinsel olarak tükendiğini, sadece kararlarını nasıl verdiğine bakarak anlayabilir misiniz? Sensör yok, anket yok, herhangi bir cihaz yok. Sadece davranış.

İşte BDFS tam olarak bunu yapıyor. Hadi başlayalım."

**🔑 Vurgu Noktası:** Soruyu sorduktan sonra 1-2 saniyelik bir duraklama yapın. *"Sadece davranış"* cümlesini yavaş ve net söyleyin — bu sizin kancınız.

---

### 🟦 SLAYT 2 — Görünmez Tehdit
**Konuşmacı: ZEYNAL**
**Süre: ~50 saniye**

**Söylenecekler:**

"Peki gerçek problem nedir? Karar yorgunluğu hem gerçek, hem ölçülebilir, hem de tehlikeli bir olgudur. Araştırmalar, kritik hataların önemli bir bölümünü — cerrahide, havacılıkta, finansal işlemlerde — tespit edilemeyen bilişsel tükenmişliğe bağlıyor.

Ama asıl can sıkıcı olan şu: Ortadaki karta bakın — pasif ve ölçeklenebilir çalışan, gerçek zamanlı *sıfır* araç var. Elimizdekiler EEG başlıkları, biyometrik sensörler ve vardiya sonu anketler — bunların hepsi ya çok pahalı, ya çok müdahaleye açık, ya da anı yakalayamayacak kadar yavaş.

Karar yorgunluğu kendini ilan etmiyor. Ve tam da bu yüzden bu kadar tehlikeli."

**🔑 Vurgu Noktası:** Üç karta sırayla işaret edin. *"Sıfır gerçek zamanlı araç"* derken durun — izleyicinin bunu sindirmesine izin verin.

---

### 🟦 SLAYT 3 — Neden Önemli
**Konuşmacı: ZEYNAL**
**Süre: ~45 saniye**

**Söylenecekler:**

"Ve bu dar bir alan problemi değil. Slayttaki dört alana bakın. Beş saatlik bir ameliyattaki cerrah. Karmaşık bir iniş yapan pilot. Otoyolda saatlerce giden bir sürücü. Ya da günün otuzuncu davasını dinleyen bir hâkim.

Bu insanların hepsi dışarıdan uyanık görünüyor olabilir — ama bilişsel olarak son demlerini yaşıyorlar.

Araştırmalar şunu gösteriyor: Hâkimler öğleden sonra koşullu tahliye kararlarını belirgin biçimde daha az onaylıyor. Davalar değişmediği halde. Sadece karar kalitesi düşüyor.

Bunu erken yakalamak sadece performansı artırmıyor — hayat kurtarıyor."

**🔑 Vurgu Noktası:** Dört karta tek tek işaret edin. Son cümleyi yavaş ve güçlü bir şekilde söyleyin.

---

### 🟦 SLAYT 4 — BDFS Çözümü
**Konuşmacı: ZEYNAL**
**Süre: ~40 saniye**

**Söylenecekler:**

"BDFS bu problemi üç temel tasarım ilkesiyle çözüyor. Birincisi: *müdahalesiz* — sistemimiz yalnızca davranışsal sinyalleri okuyor. Hiçbir donanım gerekmiyor. İkincisi: *gerçek zamanlı* — tahminler asenkron API üzerinden on milisaniyenin altında dönüyor. Üçüncüsü: *açıklanabilir* — her tahmin, hangi davranışsal sinyalin uyarıyı tetiklediğini gösteren SHAP değerleriyle birlikte geliyor.

Sağda tüm pipeline'ı görebilirsiniz — web arayüzünden FastAPI backend'e, XGBoost modelinden SHAP çıktısına kadar.

Şimdi sözü Alperen'e bırakıyorum. Veriyi ve teknik mimariyi size o anlatacak."

**🔑 Vurgu Noktası:** *Müdahalesiz, gerçek zamanlı, açıklanabilir* derken üç parmak kaldırın.

> 🔄 **DEVİR CÜMLESİ:**
> *"Alperen, sana bırakıyorum."*

---

### 🟩 SLAYT 5 — Veri & Özellik Mühendisliği
**Konuşmacı: ALPEREN**
**Süre: ~55 saniye**

> *Zeynal hafifçe geri çekilir. Alperen öne geçer veya pointer'ı alır.*

---

**Söylenecekler:**

"Teşekkürler Zeynal. Her şeyin temelindeki veriden bahsedelim.

Eğitim setimiz 150.000 davranışsal deneme kaydından oluşuyor — 99.000 yorulmamış ve 50.000 yorgun karar örneği. Her kayıt, üç kavramsal grupta 19 mühendislik özelliği içeriyor.

İlk grup *temporal özellikler* — son beş ve on denemede seçim tutarsızlığı gibi metrikleri kapsıyor. Sağdaki dağılım grafiğinde görebileceğiniz gibi, bu özellikler yorgun ve yorulmamış denekler arasında dramatik biçimde farklılaşıyor.

İkinci grup *bilişsel özellikler* — tepki sürelerini psikolojik bileşenlerine ayırmak için Drift Diffusion Modeli uyguladık: drift rate, karar sınırı ve karar dışı süre.

Üçüncüsü *davranışsal bağlam* — oturum pozisyonu, görev karmaşıklığı, doğruluk azalma hızı.

Bu 19 özellik bir araya gelince bilişsel durumun zengin, çok boyutlu bir parmak izini elde ediyoruz."

**🔑 Vurgu Noktası:** Sağdaki dağılım grafiğine işaret edin. Özellikle rolling_incon_5 paneline dikkat çekin.

---

### 🟩 SLAYT 6 — Model Değerlendirme
**Konuşmacı: ALPEREN**
**Süre: ~60 saniye**

**Söylenecekler:**

"Beş farklı sınıflandırıcıyı aynı koşullar altında eğitip değerlendirdik. Tablodaki sonuçlar açık: kazanan XGBoost — ROC-AUC 0.967, F1-Score 0.866.

Ama neden XGBoost'un kazandığını açıklamak istiyorum — çünkü bu sadece bir rakam değil. Davranışsal veri son derece doğrusal olmayan bir yapıya sahip. Yorgunluk tek bir özellikten değil, özellik kombinasyonlarından ve eşik değerlerinden ortaya çıkıyor. Ağaç tabanlı modeller bunu doğal olarak yakalıyor. Lojistik Regresyon, 0.948 AUC ile rekabetçi kalsa da daha karmaşık etkileşim örüntülerini modelleyemedi.

SVM ve KNN? Ciddi biçimde geride kaldılar. SVM yalnızca 0.771 AUC'ye ulaşabildi — XGBoost'un neredeyse 20 puan altında. Sağdaki ROC eğrisi bu farkı görsel olarak ortaya koyuyor.

XGBoost'u seçmemizin asıl nedeni hassasiyet-duyarlılık dengesini iyi kurması — yüzde 91.6 hassasiyet, gerçek bir dağıtımda sahte uyarıları minimumda tutuyor. Bu kritik."

**🔑 Vurgu Noktası:** *"Kazanan açık"* derken XGBoost satırına dokunun. ROC grafiğini gösterirken çizimlerin nasıl ayrıştığını anlatın.

---

### 🟩 SLAYT 7 — Sistem Pipeline
**Konuşmacı: ALPEREN**
**Süre: ~50 saniye**

**Söylenecekler:**

"Sistemin uçtan uca nasıl çalıştığını göstereyim — altı temiz aşama.

Kullanıcı web arayüzünü açıyor ve 19 davranışsal kaydırıcıyı ayarlıyor ya da hazır profil yüklüyor. Gönderimde bu değerler JSON olarak FastAPI backend'e HTTP POST ile iletiliyor.

Üçüncü adım kritik bir mühendislik kararı: ön işleme. StandardScaler'ı *yalnızca eğitim verisine* fit edip serileştiriyoruz ve API başlangıcında bir kez yüklüyoruz. Bu küçük bir detay değil — scaler'ı her tahmin anında yeniden fit ederseniz tahminleriniz anlamsız olur. Bunu zor yoldan öğrendik.

Dördüncü adım: XGBoost tahmini — model sınıf ve olasılık döndürüyor. Beşinci adım: SHAP hesabı — özellik katkıları yanıta ekleniyor. Altıncı adım: arayüz göstergeci, kararı ve üst faktörleri render ediyor — toplamda on milisaniyenin altında.

Alttaki not temel tasarım ilkemiz: her üretim tahmini, doğrulamada test ettiğimizle matematiksel olarak özdeş."

**🔑 Vurgu Noktası:** Altı karta sırayla işaret edin. Scaler konusunda yavaşlayın — bu güçlü bir güvenilirlik sinyali.

---

### 🟩 SLAYT 8 — Güçlü Backend Entegrasyonu
**Konuşmacı: ALPEREN**
**Süre: ~45 saniye**

**Söylenecekler:**

"Backend Python'un en hızlı modern web framework'lerinden biri olan FastAPI üzerine kurulu — asenkron I/O ile çalışıyor, yani eş zamanlı istekleri kuyruklamadan işleyebiliyor.

API beş temiz endpoint sunuyor: health check, schema, /predict'te tekil tahmin, yüksek iş hacmi için /predict/batch ve örnek veri için /example. Sağdaki Swagger UI otomatik oluşturuldu — bu sadece dokümantasyon değil, geliştirme sürecinde kullandığımız canlı bir test arayüzü.

Her giriş Pydantic modelleriyle doğrulanıyor — eksik ya da sınır dışı bir alan olursa istek modele ulaşmadan yapılandırılmış hata mesajıyla reddediliyor. Her tahmin yanıtı SHAP katkılarını da içeriyor — yani API sadece bir tahmin motoru değil, tam bir açıklanabilirlik servisi.

Tamam — yeterince anlattık. Çalışan sistemi gösterelim."

**🔑 Vurgu Noktası:** Swagger ekran görüntüsüne doğrudan atıfta bulunun. Son cümleyi hafif bir gülümsemeyle, enerjik söyleyin — demo geçişiniz bu.

---

### 🟩 SLAYT 9 — CANLI DEMO
**Konuşmacı: ALPEREN**
**Süre: ~90 saniye**

> *Tarayıcıya geçin. Bu slayt arka planda görünmeye devam eder.*

---

*(Tam demo scripti için aşağıdaki Bölüm 4'e bakın.)*

---

### 🟩 SLAYT 10 — Kullanıcı Arayüzü
**Konuşmacı: ALPEREN**
**Süre: ~40 saniye**

> *Demo sonrası slayt güvertesine dönün.*

---

**Söylenecekler:**

"Az önce gördüğünüz tam arayüzdü. Bilinçli olarak aldığımız birkaç tasarım kararını vurgulayayım.

Koyu glassmorphic estetik sadece görsel bir tercih değil — hassasiyet ve odaklanmayı simgeliyor. Basit/Uzman mod geçişi, teknik bilgisi olmayan bir operatörün sıfır eğitimle kullanabilmesini sağlarken araştırmacılara tüm 19 ham değer ve DDM parametrelerine erişim sunuyor.

Sağ üst köşedeki tooltip ekran görüntüsü? Her kaydırıcıda var. Sade Türkçe açıklamalar. Jargon yok. Çünkü yalnızca uzmanların kullanabildiği bir sistem gerçek hayatta işe yaramaz.

Olasılık göstergesinin gerçek zamanlı animasyon yaptığına dikkat edin — her kaydırıcı değişimi canlı bir API çağrısı tetikliyor. Sıfır gecikme."

**🔑 Vurgu Noktası:** Tooltip ekran görüntüsüne işaret edin. Canlı API çağrısı özelliğini vurgulayın.

> 🔄 **DEVİR CÜMLESİ:**
> *"Şimdi sözü Zeynal'a bırakıyorum — SHAP analizimizin ne ortaya koyduğunu anlatacak. Bu aslında tüm projenin en şaşırtıcı bulgusuydu."*

---

### 🟦 SLAYT 11 — Sonuçlar & Açıklanabilirlik (SHAP)
**Konuşmacı: ZEYNAL**
**Süre: ~45 saniye**

> *Zeynal tekrar öne geçer.*

---

**Söylenecekler:**

"Teşekkürler Alperen. Ve evet — bizi asıl şaşırtan kısım bu.

SHAP analizini çalıştırdığımızda DDM bilişsel parametrelerinin — drift rate, karar sınırı — baskın çıkmasını bekliyorduk. Bunlar iyi doğrulanmış bir psikoloji modelinden türetiliyor. En güçlü sinyalin bunlar olacağını düşündük.

Ama grafiğe bakın. SHAP önem sıralamasında ilk iki özellik rolling_incon_5 ve rolling_incon_10 — saf temporal tutarsızlık. Birinin *son beş ve on denemesinde* ne kadar tutarsız seçimler yaptığı. Bunlar tüm statik bilişsel metrikleri geride bıraktı.

Ablasyon çalışmamız da bunu doğruladı: Tüm DDM özelliklerini çıkarmak AUC'u yüzde 0.1'den az düşürüyor. Ama temporal özellikleri çıkarmak performansı dört puanın üzerinde düşürüyor.

Bulgu şu: Yorgunluk tek bir anda ortaya çıkmıyor. *Zamanla biriken düzensiz davranışlarla* kendini ele veriyor. Bu çalışmanın temel bilimsel katkısı budur."

**🔑 Vurgu Noktası:** SHAP grafiğinde ilk iki özelliğe işaret edin. Son cümle teziniz — net ve güvenle söyleyin.

---

### 🟦 SLAYT 12 — Zorluklar & Çözümler
**Konuşmacı: ZEYNAL**
**Süre: ~30 saniye**

**Söylenecekler:**

"Hiçbir proje sorunsuz ilerlemez ve biz bunu açıkça paylaşmak istiyoruz.

En büyük teknik zorluk veri ölçekleme uyumsuzluğuydu. API'nin erken sürümlerinde scaler her istekte yeniden fit ediliyordu — üretim tahminlerini tamamen anlamsız kılıyordu. Çözüm basitti ama kritikti: fit edilmiş scaler'ı eğitim sonunda serileştir, API başlangıcında bir kez yükle.

İkinci zorluk kullanıcı deneyimiydi. 'ez_drift_rate' veya 'drift_boundary_ratio' gibi 19 ham özellik değişkeni çoğu kullanıcıya hiçbir şey ifade etmiyor. Çözümümüz Basit/Uzman geçişi — karmaşıklığı gizlemeden soyutlamak.

Bunlar sadece hatalar değildi. Araştırma ortamından gerçek ürüne geçişin dersleriydiler."

**🔑 Vurgu Noktası:** Son cümleyi kararlılıkla söyleyin — araştırma ile dağıtım arasındaki farkı anladığınızı gösteriyor.

---

### 🟦 SLAYT 13 — Gelecek Planlar
**Konuşmacı: ZEYNAL**
**Süre: ~20 saniye**

**Söylenecekler:**

"Geleceğe hızlıca bakalım — kısa vadede pasif girdi entegrasyonu: klavye telemetrisi, fare dinamikleri, web kamerası tabanlı göz takibi. Yeni donanım gerekmez, sadece daha zengin davranışsal akışlar.

Orta vadede kullanıcı başına kişiselleştirilmiş taban çizgileri ve uyarlanabilir eşik değerleri. Uzun vadede gerçek sağlık ve havacılık ortamlarında klinik pilot çalışmalar — ve hassas verileri merkezi sunucuda toplamadan modeli geliştirebilmek için federated learning.

Altyapı hazır. Sadece daha iyi veriye ihtiyacı var."

**🔑 Vurgu Noktası:** Hızlı ve ileriye dönük tutun. Vizyon çiziyorsunuz, plan savunmuyorsunuz.

---

### 🟦 SLAYT 14 — Sonuç & Sorular
**Konuşmacı: ZEYNAL + ALPEREN**
**Süre: ~25 saniye**

**ZEYNAL şöyle başlar:**

"Özetle: Behavioral Decision Fatigue Scoring — sadece davranışsal veri kullanarak bilişsel karar yorgunluğunu gerçek zamanlı tespit eden, ROC-AUC 0.967 ve on milisaniyenin altında gecikmeyle çalışan üretime hazır bir sistem inşa ettik.

Temel bilimsel bulgu: temporal davranışsal tutarsızlık, herhangi bir statik bilişsel metrikten daha güçlü bir yorgunluk sinyali taşıyor.

Sözü Alperen'e bırakmadan önce — bu projeye başından beri katkı koyan danışmanımız Asistan Profesör Doktor Denizhan Demirkol'a içtenlikle teşekkür etmek istiyoruz. Rehberliği ve desteği olmadan bu sonuçlara ulaşamazdık."

**ALPEREN ekler:**

"Ayırdığınız zaman için teşekkür ederiz. Sorularınızı almaktan memnuniyet duyarız."

**🔑 Vurgu Noktası:** Dr. Demirkol'a teşekkür samimi ve sıcak olmalı — eğer odadaysa göz teması kurun. Alperen'in son cümlesi sakin ve özgüvenli olmalı, aceleci değil.

---

## 🔄 BÖLÜM 3 — TÜM GEÇİŞLER

| # | Kimden | Kime | Devir Cümlesi |
|---|---|---|---|
| 1 | Zeynal (Slayt 4) | Alperen (Slayt 5) | *"Şimdi sözü Alperen'e bırakıyorum — veriyi ve teknik mimariyi size o anlatacak."* |
| 2 | Alperen (Slayt 10) | Zeynal (Slayt 11) | *"Şimdi sözü Zeynal'a bırakıyorum — SHAP analizimizin ne ortaya koyduğunu anlatacak. Bu aslında tüm projenin en şaşırtıcı bulgusuydu."* |
| 3 | Zeynal (Slayt 13) | Her ikisi (Slayt 14) | *(Zeynal sonucu anlatır, Alperen soru-cevabı kapatır — doğal çift kapanış, açık sözlü devire gerek yok)* |

> 💡 **Taktik:** Devir alırken hemen başlamayın. Bir nefes kadar bekleyin. Bu izleyiciye kontrol ve özgüven sinyali verir.

---

## 🖥️ BÖLÜM 4 — CANLI DEMO SCRIPTİ

**Konuşmacı: ALPEREN**
**Süre: ~90 saniye**
**Başlangıç noktası:** Slayt 9 ekranda. Alperen tarayıcıyı açar.

---

### Adım 1 — Uygulamayı Aç
> *BDFS web uygulaması açık tarayıcıya geç.*

**Söylenecekler:**
> *"Tamam, bu canlı uygulama — şu an çalışıyor, FastAPI backend'imize bağlı. Sağ üst köşede model rozetini görebilirsiniz: XGBoost, F1 0.866, AUC 0.967 ve yeşil 'Bağlı' göstergesi — API aktif."*

---

### Adım 2 — Yorgun Profil Yükle
> *Sol üstteki "Load Fatigued Profile" butonuna tıklayın.*

**Söylenecekler:**
> *"Test setimizden gerçek bir örnek olan yorgun profilimizi yüklüyorum. Sağdaki göstergeyi izleyin."*

> *[1 saniye bekleyin, gösterge animasyonu tamamlanana kadar.]*

> *"Yüzde 87. Sistem bu durumu orta-yüksek güvenle YORGUN olarak sınıflandırdı. Göstergenin altındaki üst katkıda bulunan özelliklere bakın: Kısa Tutarsızlık, Uzun Tutarsızlık ve Karar Değiştirme — tam olarak SHAP analizimizin öngördüğü özellikler."*

---

### Adım 3 — Tooltip Göster
> *"Reaction Speed" veya "Short Inconsistency" yanındaki ⓘ simgesinin üzerine gelin.*

**Söylenecekler:**
> *"Her kaydırıcının sade bir açıklaması var. Bu şunu söylüyor: 'Deneme başına ortalama tepki süresi — yüksek değer yavaş tepkiyi ifade eder.' ML bilgisi sıfır olan biri de ne baktığını anlayabiliyor."*

---

### Adım 4 — Kaydırıcıyı Manuel Ayarla
> *Expert moda geçin. Ardından "Short Inconsistency" (rolling_incon_5) kaydırıcısını yavaşça sola çekin.*

**Söylenecekler:**
> *"Şimdi tutarsızlık skorunu düşürürsem ne olduğuna bakın — daha tutarlı bir karar vericiyi simüle ediyorum. Gösterge gerçek zamanlı değişiyor. Her kaydırıcı hareketi backend'e canlı bir API çağrısı yapıyor. Gecikme yok, batch işleme yok."*

> *[Kaydırıcıyı çekin. Gösterge animasyonunu bekleyin.]*

> *"Yüzde 40'a indi. Aynı oturum, aynı her şey — sadece daha tutarlı seçimler. Sistem bunu anında fark etti."*

---

### Adım 5 — Alert Profil Yükle (Süre varsa)
> *Üstteki "Load Alert Profile" butonuna tıklayın.*

**Söylenecekler:**
> *"Bir de tam tersi — uyanık, yorulmamış profil. Gösterge tek haneli rakamlara düşüyor. Sistem: yorgun değil diyor. Modelin çalıştığı dinamik aralık bu."*

---

### Adım 6 — Slaytlara Dön
> *Sunum güvertesine geri dönün.*

**Söylenecekler:**
> *"BDFS böyle çalışıyor. Az önce gördüğünüz tüm döngü — girişten tahmine, tahminden açıklamaya — sunucu tarafında on milisaniyenin altında gerçekleşiyor. Slaytlara dönelim."*

---

### ⚠️ ACİL DURUM SATIRLARI — Bir Şeyler Ters Giderse

| Problem | Ne Söylenecek |
|---|---|
| API bağlı değil / "Disconnected" yazıyor | *"Canlı backend şu an yanıt vermiyor — demo ortamlarında bazen böyle oluyor. Kayıtlı bir walkthrough'um var ama arayüzü yine de gösterebilirim, çünkü önemli olan mimari."* |
| Tarayıcı açılmıyor | *"Tarayıcı biraz inatçı davranıyor — bir sonraki slayttaki ekran görüntülerine geçelim, tam olarak ne göreceğinizi orada yakaladık."* |
| Profil yüklenince değerler yanlış çıkıyor | *"Değerler biraz kaymış gibi — sıfırlayıp profili tekrar yükleyeyim."* *(Sakin bir şekilde Reset'e, sonra Fatigued Profile'a tıklayın)* |
| Gösterge animasyon yapmıyor | *"Animasyon takıldı gibi görünüyor ama olasılık değeri güncellendi. Backend tahminleri doğru döndürüyor, bu frontend render zamanlamasıyla ilgili küçük bir şey."* |
| Demo sırasında soru sorulursa | *"Güzel soru — şu adımı tamamlayayım, hemen döneceğim."* |

> 💡 **Genel kural:** Aşırı özür dilemeyin. Kabul edin, pivot yapın, devam edin. İzleyici sizin enerjinizi takip eder, teknik aksaklıklarınızı değil.

---

## 📌 SUNUM GÜNÜ SON KONTROL LİSTESİ

- [ ] API backend salona girmeden önce çalışıyor
- [ ] Tarayıcı uygulamaya önceden açık (sadece URL değil, sayfa yüklenmiş)
- [ ] "Load Fatigued Profile" son 30 dakika içinde test edildi
- [ ] Slayt güvertesi **Presenter View** modunda (notlarınızı siz görebilirsiniz, izleyici göremez)
- [ ] Her iki konuşmacı hangi slaytta devreye girdiğini biliyor
- [ ] Zeynal slayt 1–4 ve 11–14'ü zihinsel olarak prova etti
- [ ] Alperen slayt 5–10 ve demoyu zihinsel olarak prova etti
- [ ] En az bir kez tam zamanlı prova yapıldı

---

*Alperen Sümeroğlu (231805023) · Zeynalabidin Ramazanzade (231805121)*
