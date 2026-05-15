import os
import glob
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
import warnings

# Ignore librosa warnings about short audio files
warnings.filterwarnings("ignore")

def extract_features(file_path):
    # Ses dosyasını yükle (zaten 16000 Hz ve mono formatında)
    y, sr = librosa.load(file_path, sr=None)
    
    if len(y) == 0:
        raise ValueError("Audio length is 0 (completely silent or corrupt file)")
        
    # Ortak hesaplamalar
    stft = np.abs(librosa.stft(y))
    
    # 1. Amplitude Envelope Mean
    frame_length = 2048
    hop_length = 512
    amplitude_envelope = np.array([max(y[i:i+frame_length]) for i in range(0, len(y), hop_length)])
    amp_env_mean = np.mean(amplitude_envelope) if len(amplitude_envelope) > 0 else 0.0
    
    # 2. RMS Mean
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    
    # 3. ZCR Mean
    zcr = librosa.feature.zero_crossing_rate(y=y)
    zcr_mean = np.mean(zcr)
    
    # 4. Spectral Centroid Mean
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    sc_mean = np.mean(sc)
    
    # 5. Spectral Bandwidth Mean
    sban = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    sban_mean = np.mean(sban)
    
    # 6. Spectral Contrast Mean
    scon = librosa.feature.spectral_contrast(S=stft, sr=sr)
    scon_mean = np.mean(scon)
    
    # 7. MFCCs13Mean
    mfccs13 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs13_mean = np.mean(mfccs13)
    
    # 8. delMFCCs13 ve del2MFCCs13
    delta_mfccs13 = librosa.feature.delta(mfccs13)
    delta2_mfccs13 = librosa.feature.delta(mfccs13, order=2)
    del_mfccs13_mean = np.mean(delta_mfccs13)
    del2_mfccs13_mean = np.mean(delta2_mfccs13)
    
    # 9. MFCCs20 Mean
    mfccs20 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs20_mean = np.mean(mfccs20)
    
    return [
        amp_env_mean, rms_mean, zcr_mean, sc_mean, sban_mean, scon_mean,
        mfccs13_mean, del_mfccs13_mean, del2_mfccs13_mean, mfccs20_mean
    ]

def feature_extraction_pipeline():
    base_dir = r"C:\Users\ASUS\Desktop\aml_project"
    csv_path = os.path.join(base_dir, "raw_data", "donateacry-corpus_features_final.csv")
    audio_dir = os.path.join(base_dir, "standardized_audio")
    
    print("Metadata CSV yükleniyor...")
    df = pd.read_csv(csv_path)
    
    # Dosya adını orijinal yoldan çıkartarak bir eşleştirme (mapping) sözlüğü oluşturalım
    df['original_filename'] = df['Cry_Audio_File'].apply(lambda x: Path(x).name)
    df['stem'] = df['Cry_Audio_File'].apply(lambda x: Path(x).stem)
    
    # Uzantı bağımsız eşleştirme yapabilmek için stem üzerinden eşleştirme
    label_map = dict(zip(df['stem'], df['Cry_Reason']))
    
    # standardized_audio klasöründeki .wav dosyalarını al
    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    
    features_list = []
    labels_list = []
    
    total_files = len(audio_files)
    print(f"standardized_audio klasöründe toplam {total_files} fiziksel dosya bulundu.\n")
    
    match_count = 0
    extracted_count = 0
    
    for idx, file_path in enumerate(audio_files, 1):
        filename = Path(file_path).name
        stem = Path(file_path).stem
        
        # Orijinal uzantısı farklı olan dosyalar için (.mp3 -> .wav vs.) 
        # ismin kökü (stem) üzerinden metadata eşleşmesi arıyoruz.
        if stem in label_map:
            match_count += 1
            label = label_map[stem]
            
            print(f"[{idx}/{total_files}] Özellikler çıkarılıyor: {filename}...")
            try:
                features = extract_features(file_path)
                features_list.append([filename] + features + [label])
                labels_list.append(label)
                extracted_count += 1
            except Exception as e:
                print(f"[{idx}/{total_files}] HATA ({filename}): {e}")
        else:
            print(f"[{idx}/{total_files}] Eşleşme BULUNAMADI, atlanıyor: {filename}")
    
    columns = [
        "filename", "Amplitude_Envelope_Mean", "RMS_Mean", "ZCR_Mean", 
        "SC_Mean", "SBAN_Mean", "SCON_Mean", "MFCCs13Mean", 
        "delMFCCs13", "del2MFCCs13", "MFCCs20", "Cry_Reason"
    ]
    
    # Final veri çerçevesini oluştur
    feature_df = pd.DataFrame(features_list, columns=columns)
    
    # 1. CSV olarak kaydet
    out_csv = os.path.join(base_dir, "final_feature_matrix_v2.csv")
    feature_df.to_csv(out_csv, index=False)
    
    # 2. NumPy arrayleri olarak kaydet
    X = feature_df.drop(columns=["filename", "Cry_Reason"]).values.astype(np.float32)
    y = feature_df["Cry_Reason"].values.astype(np.int32)
    
    out_x_npy = os.path.join(base_dir, "X_tabular.npy")
    out_y_npy = os.path.join(base_dir, "y_labels.npy")
    
    np.save(out_x_npy, X)
    np.save(out_y_npy, y)
    
    print("\n" + "="*40)
    print("ÖZELLİK ÇIKARIM İŞLEMİ TAMAMLANDI")
    print(f"Klasördeki toplam dosya: {total_files}")
    print(f"CSV'de eşleşen dosya: {match_count}")
    print(f"Başarıyla özellikleri çıkarılan dosya: {extracted_count}")
    print("="*40)
    print("Çıktılar kaydedildi:")
    print(f" - CSV Tablosu: final_feature_matrix_v2.csv")
    print(f" - Model Girdileri (X): X_tabular.npy")
    print(f" - Etiketler (y): y_labels.npy")

if __name__ == "__main__":
    feature_extraction_pipeline()
