# Cleaned AML Dataset Report

- **Toplam örnek sayısı**: 634
- **X_images_new shape**: (634, 128, 128, 1)
- **X_tabular_new shape**: (634, 65)
- **y_labels_new shape**: 634
- **groups_new shape**: 634

## Sınıf Dağılımı
label
3    239
2    166
0    126
1     78
4     25

## Temizleme Özeti
- **Yöntem**: Tüm raw dosyalardan MD5 Hash alınarak duplicate'ler engellendi.
- **Label Çakışması**: Farklı klasörlerde aynı hash'e sahip dosyalar tespit edildi. Çoğunluk label'ı atandı.
- **Feature Düzeltmesi**: Tabular feature çıkarılırken `axis=1` kullanıldı. Boyut 10'dan 65'e çıktı.
- **Group ID**: Hash değeri doğrudan group_id olarak kullanıldı. Böylece leakage tamamen engellendi.
