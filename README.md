# 👶 Baby Cry Classification (AML Project)

## 📌 1. Proje Özeti
Bu proje, bebeklerin ağlama seslerini analiz ederek ağlama nedenlerini (Açlık, Karın Ağrısı, Gaz Çıkarma, Rahatsızlık, Yorgunluk) sınıflandırmayı amaçlayan uçtan uca (end-to-end) bir Makine Öğrenimi projesidir. 

Gerçek dünya verileri genellikle çok gürültülü ve problemlidir. Bu çalışma, sadece bir model eğitmenin ötesinde; ham, bozuk ve ciddi şekilde "veri sızıntısı" (data leakage) barındıran bir veri setinden yola çıkarak **veri temizleme**, **bölütleme (group splitting)**, **feature engineering (öznitelik çıkarımı)** ve **hyperparameter/threshold optimization** süreçleriyle sağlam, güvenilir ve üretime hazır (production-ready) bir Tabular ML Pipeline'ı inşa etmenin önemini vurgular.

---

## 🚨 2. Başlangıç Problemi
Projenin başlangıcında kurulan kompleks hibrit (CNN + LSTM) modeller bile validasyon setinde **%2 ~ %5 Accuracy** seviyesini geçemedi ve model asla öğrenemedi. Yapılan derin veri denetimi (data audit) sonucunda şu felaket senaryoları tespit edildi:

- **Çok Yüksek Veri Tekrarı (Duplication):** Veri setindeki dosyaların yaklaşık **%77'si** birbirinin aynısı (kopyalanmış dosyalar) idi.
- **Çelişkili Etiketler (Contradictory Labels):** Aynı ses dosyasının kopyası, farklı klasörlere konulup "Hungry" ve "Tired" gibi farklı etiketlerle etiketlenmişti. (Modelin kafasının karışmasının ve öğrenememesinin ana nedeni).
- **Veri Sızıntısı (Data Leakage):** Aynı sesten üretilen farklı parçalar hem Train hem de Test setine dağılmıştı.
- **Kusurlu Feature Extraction:** MFCC özellikleri çıkarılırken zaman boyutunun ortalaması yanlış bir eksende (`np.mean(mfcc)` yerine `np.mean(mfcc, axis=1)`) alınmış, bu da spektrogram bilgisini 1 boyutlu işe yaramaz bir skalere dönüştürmüştü.

---

## 🧹 3. Veri Temizleme Süreci
Problemi temelinden çözmek adına radikal bir temizlik operasyonu uygulandı:

- **MD5 Hash Deduplication:** Tüm raw audio (`.wav`) dosyalarının MD5 hash'i hesaplanıp benzersiz imzaları çıkarıldı.
- **Majority Label Stratejisi:** Aynı ses dosyasında farklı etiketler (çelişki) bulunursa, bu dosyanın klasörlerde en çok tekrar ettiği etiket (Majority Voting) baz alındı. Diğer çakışan ve kopyalanan kopyalar silindi.
- **Group ID Ataması:** Train/Test ayrımında hiçbir sesin sızmaması için `hash` imzaları **Group ID** olarak kabul edildi.
- **Sonuç:** 1,816 dosyalık kirli veri seti, **634 adet** tamamen benzersiz, doğru etiketlenmiş ses dosyasına indirgendi (`cleaned_audio_unique/`).

---

## 🧠 4. Feature Engineering
Sadece MFCC özellikleri yeterli değildi. Genişletilmiş ve V2 olarak adlandırılan yeni nesil bir özellik matrisi (`X_tabular_v2.npy`) oluşturuldu:

- **Akustik Düzeltmeler:** `axis=1` düzeltmesi ile MFCC13, Delta MFCC13, Delta2 MFCC13 ve MFCC20 özellikleri doğru matris yapısında ortalamalar alınarak çıkarıldı.
- **Spektral Özellikler:** Spectral Centroid (Parlaklık), Spectral Bandwidth, Spectral Rolloff.
- **Zaman/Enerji:** RMS (Enerji), ZCR (Sıfır Geçiş Oranı).
- **Frekans Analizi:** `librosa.yin` algoritması ile Pitch (f0) - min, max, mean, std.
- **Duration:** Sesin süresi eklendi.
- **Boyut:** Veri matrisi boyutu `65`'ten **`74` boyutlu** bir feature space'e genişletildi.

---

## 🤖 5. Model Denemeleri
Veri seti küçüldükten sonra (634 örnek) Derin Öğrenme (CNN) modelleri overfit olmaya mahkumdu. Biz de strateji değiştirip daha verimli Tabular Makine Öğrenimi modellerine yöneldik. `StratifiedGroupKFold` ile yapılan 5-Fold Cross Validation testlerinde şu modeller değerlendirildi:

- **RandomForest** 
- **ExtraTrees** *(En iyi baz performansı sergiledi)*
- **HistGradientBoosting**
- **SVC** *(Support Vector Classifier - Azınlık sınıflarda dengeleyici rol oynadı)*
- **LogisticRegression**

---

## 🏆 6. Final Model: Çift Motorlu Ensemble Pipeline
Maksimum performansa erişebilmek adına klasik bir model kullanmak yerine aşağıdaki hibrit strateji uygulandı:

1. **Veri Dengeleme (SMOTE):** Veri sızıntısını önlemek için `StratifiedGroupKFold` içindeki *sadece Train setine* SMOTE sentetik veri büyütme tekniği uygulanarak tüm sınıflar dengelendi.
2. **Soft Voting Ensemble:** Modeller `%70 ExtraTrees` ve `%30 SVC` olasılık ağırlıklarıyla (predict_proba) harmanlandı.
3. **Class-Specific Thresholding:** Grid Search ile her bir sınıf için özel probability eşik değerleri atandı. Örneğin en zor sınıf olan `Tired` sınıfının eşiği (threshold) 0.1'e indirilerek modelin bu azınlık sınıfını yakalaması sağlandı.

---

## 📊 7. Final Sonuçlar
Başlangıçtaki %2'lik doğruluk oranından, Tabular Pipeline ve optimizasyonlar sonucunda ulaşılan final değerler (Hold-out Test Seti üzerinde):

- **Macro F1 Skoru:** `0.4999`
- **Macro Recall:** `0.4697`
- **Accuracy:** `0.5276`

**Dikkat Çeken Sıçramalar:**
- Başlangıçta performansı **%0** olan `Tired` (Yorgunluk) sınıfının Recall değeri **%40**'a (`0.4000`) çıkarıldı.
- `Belly Pain` sınıfının Recall değeri SMOTE sonrası **%56**'ya (`0.5600`) ulaştı.
- Çoğunluk sınıfı olan `Hungry` **%77** bandında istikrarlı tutuldu.

---

## 📁 8. Proje Yapısı

```bash
aml_project/
├── raw_data/                 # Orjinal kirli ve ham ses veri setleri
├── cleaned_audio_unique/     # Deduplication sonrası elde edilen temiz wav dosyaları
├── outputs_for_colab/        # İlk iterasyon başarısız CNN çıktıları
├── outputs_for_colab_clean/  # Yeniden inşa edilen temiz V1 ve V2 matrisleri
├── results_final_model/      # Eğitilmiş pkl objeleri, eşikler ve metrik analizleri
├── scripts/                  # Tüm data pipeline scriptleri
│   ├── rebuild_dataset.py    # Deduplication ve veri temizliği
│   └── validate_outputs.py   # Tensor shape doğrulamaları
├── build_final_model.py      # Son eğitim pipeline'ını koşturan script
└── inference.py              # Eğitilmiş modeli canlı (production) ortamda çalıştıran script
```

---

## ⚙️ 9. Nasıl Çalıştırılır

**1. Veri Temizliği ve Matris İnşası:**
```bash
python scripts/rebuild_dataset.py
```

**2. Gelişmiş Özellik Çıkarımı (Feature Extraction V2):**
```bash
python outputs_for_colab_clean/extract_expanded_features.py
```

**3. Final Model Eğitimi (SMOTE + Ensemble + Thresholding):**
```bash
python outputs_for_colab_clean/build_final_model.py
```

**4. Canlı Tahminleme (Inference):**
Eğitilen model kullanılarak ham bir `.wav` dosyası üzerinde tahmin yapmak:
```bash
python inference.py "raw_data/BABY CRY/Hungry/test_audio.wav"
```

---

## 🔮 10. Gelecek Çalışmalar
- **Daha Fazla Veri:** SMOTE işe yarasa da, model kapasitesinin artırılmasının önündeki en büyük engel 634 örneklik küçük dataset boyutudur.
- **Audio Augmentation:** Sadece SMOTE (Tabular sentetik) yerine, ham audio üzerinde `Time Stretch`, `Pitch Shift` gibi veri çoğaltmaları yapılabilir.
- **Manual Relabeling:** Hala seslerde yanlış etiketlendiği düşünülen çakışmalar uzman bir pediatrist eşliğinde elden geçirilebilir.
- **Deep Learning Yeniden Deneme:** Dataset 5,000+ örneğe çıktığında, spectrogram özellikleriyle ResNet tabanlı bir model denenebilir.

---

## 🧪 11. Teknik Notlar
- **Group-Based Split:** Aynı ortamdan (veya aynı bebekten) gelen verilerin Train'de olup Test'te model başarısını sahte (yapay) şekilde artırmasını engellemek için her zaman `StratifiedGroupKFold` veya `GroupShuffleSplit` kullanılmalıdır. Bu projede grup ID olarak "Dosya MD5 Hasherı" baz alındı.
- **Neden Accuracy Yeterli Değil?:** Veri seti %80 oranında `Hungry` (Açlık) içeriyordu. Sürekli Hungry tahmin eden aptal bir modelin Accuracy'si %80 çıkabilirdi. Gerçek başarıyı ölçmek için `Macro F1` ve `Per-Class Recall` odaklı ilerlenmiştir.

---

## 📌 12. Sonuç
Bu proje, makine öğreniminde **"Garbage in, Garbage out"** (Çöp girerse, çöp çıkar) ilkesinin mükemmel bir kanıtı niteliğindedir. Aylarca üzerinde hiperparametre optimizasyonu yapılan kompleks Deep Learning modellerinin çözemediği bir problem; **doğru data audit, hash-bazlı temizlik ve iyi tasarlanmış özellik çıkarımı (feature engineering)** ile ağaç tabanlı (tree-based) klasik makine öğrenimi modelleri tarafından başarıyla çözülmüştür. 

En kritik öğrenilen ders: *Model mimariniz ne kadar iyi olursa olsun, veri setinizin bütünlüğü projeyi vezir de eder, rezil de.*
