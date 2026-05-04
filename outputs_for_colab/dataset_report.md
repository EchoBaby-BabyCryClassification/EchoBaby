# AML Project - Dataset Final Report

- **Toplam örnek sayısı**: 457
- **X_images shape**: (457, 128, 128, 1)
- **X_tabular shape**: (457, 10)
- **y_labels shape**: (457,)
- **groups shape**: (457,)

## Sınıf Dağılımı
Cry_Reason
3    382
2     27
4     24
0     16
1      8

## Grup (Subject) İstatistikleri
- **Grup sayısı**: 221
- **Grup başına ortalama örnek sayısı**: 2.07

### En büyük 20 grup
group_id
D6CDA191-4962-4308-9A36-46D5648A95ED    13
999BF14B-E417-4B44-B746-9253F81EFE38    10
3BB2445A-9AE9-4DC8-9A2E-86C696FFA405     8
40A4C760-FACF-4911-B0A6-22ECCC3AB18D     8
643D64AD-B711-469A-AF69-55C0D5D3E30F     8
8F4DBF5A-B504-4452-B6AC-94E646A6A052     8
1309B82C-F146-46F0-A723-45345AFA6EA8     7
AAA57DBD-7B88-454A-963E-6FAA2F4ED4E7     7
10A40438-09AA-4A21-83B4-8119F03F7A11     6
549A46D8-9C84-430E-ADE8-97EAE2BEF787     6
7E4B9C14-F955-4BED-9B03-7F3096A6CBFF     6
3030D0E9-0C9B-4616-9B83-F10DC2DEDBFD     5
9CFD61B9-BF13-406D-8B2F-F73CFAAF25CB     5
C421C6FE-DFEE-4080-8AEA-848E7CE4756B     5
C6FD9D60-0FA7-44C0-B3CE-0192527D7B81     5
090C15A8-5406-4EA5-97A3-81F6527227C0     4
4BE720CE-A5E5-4A48-930F-A212F8A239F6     4
50DC5FBC-AC95-44F2-B55F-E4E5FB03AC96     4
64ACB345-A61E-4EF3-A5A6-CF83C04B83F1     4
719BB382-A592-46B7-82D2-8B4A625263B7     4

## Leakage ve Split Analizi
Aynı `group_id`'ye (UUID) sahip örnekler, orijinal ses dosyasından bölünen veya aynı kayıt cihazından aynı seansta kaydedilen farklı ses dosyalarıdır. Bu grupları böldüğümüzde Data Leakage tamamen engellenecektir. `groups.npy` bu yüzden oluşturuldu.

**Kullanılan group_id çıkarım mantığı**: `[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-...` formatındaki UUID yapısı her bir dosya isminin başından regex ile çıkarılarak Subject/Session kimliği elde edilmiştir.
