import os
import pandas as pd
import numpy as np
import librosa
import warnings

# Gereksiz uyarıları kapatalım
warnings.filterwarnings("ignore")

def generate_mel_spectrograms():
    base_dir = r"C:\Users\ASUS\Desktop\aml_project"
    csv_path = os.path.join(base_dir, "final_feature_matrix_v2.csv")
    audio_dir = os.path.join(base_dir, "standardized_audio")
    
    # CSV dosyasının varlığını kontrol et
    if not os.path.exists(csv_path):
        print(f"HATA: {csv_path} bulunamadı.")
        return
        
    print("CSV metadata okunuyor...")
    df = pd.read_csv(csv_path)
    
    X_images = []
    
    # CNN için Mel-Spektrogram Parametreleri
    sr = 16000
    n_mels = 128
    hop_length = 512
    n_fft = 2048
    
    # librosa.melspectrogram'da center=True varsayılan olduğu için, 
    # (128, 128) boyutunda bir matris elde etmek adına ses uzunluğunu tam olarak ayarlıyoruz:
    # time_steps = 1 + (len(y) / hop_length). 128 = 1 + (len(y) / 512) => len(y) = 127 * 512 = 65024
    target_length = (n_mels - 1) * hop_length 
    
    total_files = len(df)
    print(f"Toplam {total_files} dosya işlenecek...\n")
    
    for idx, row in df.iterrows():
        filename = row['filename']
        file_path = os.path.join(audio_dir, filename)
        
        try:
            # Sesi yükle
            y, _ = librosa.load(file_path, sr=sr)
            
            # Ses boyutunu sabitle (Kısaysa sonuna sıfır ekler, uzunsa kırpar)
            y_fixed = librosa.util.fix_length(y, size=target_length)
            
            # Mel-Spektrogram çıkar
            mel_spec = librosa.feature.melspectrogram(
                y=y_fixed, 
                sr=sr, 
                n_mels=n_mels, 
                n_fft=n_fft, 
                hop_length=hop_length
            )
            
            # CNN'ler için logaritmik (dB) skalası her zaman daha iyi sonuç verir
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Min-Max Normalizasyonu (0 - 1 arası)
            min_val = mel_spec_db.min()
            max_val = mel_spec_db.max()
            
            if max_val - min_val > 0:
                mel_spec_norm = (mel_spec_db - min_val) / (max_val - min_val)
            else:
                mel_spec_norm = mel_spec_db - min_val
                
            # Kanal boyutunu ekle => (128, 128, 1) - Gri tonlamalı resim formatı
            mel_spec_img = np.expand_dims(mel_spec_norm, axis=-1)
            
            X_images.append(mel_spec_img)
            
            if (idx + 1) % 50 == 0 or (idx + 1) == total_files:
                print(f"[{idx + 1}/{total_files}] İşlendi: {filename}")
                
        except Exception as e:
            print(f"Hata oluştu ({filename}): {e}")
            
    # Listeyi NumPy array'e çevir
    X_images = np.array(X_images, dtype=np.float32)
    
    print("\n" + "="*40)
    print(f"Tüm Mel-Spektrogramlar Üretildi!")
    print(f"X_images.npy Boyutu: {X_images.shape}")
    
    # Boyut doğrulama
    if len(X_images) == total_files and X_images.shape[1:] == (128, 128, 1):
        out_path = os.path.join(base_dir, "X_images.npy")
        np.save(out_path, X_images)
        print(f"Başarıyla Kaydedildi: {out_path}")
    else:
        print("UYARI: Matris boyutu beklendiği gibi değil. Dosya kaydedilmedi!")
    print("="*40)

if __name__ == "__main__":
    generate_mel_spectrograms()
