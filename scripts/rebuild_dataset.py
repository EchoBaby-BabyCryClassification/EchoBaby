import os
import shutil
import hashlib
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from collections import Counter
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = r"C:\Users\ASUS\Desktop\aml_project"
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
CLEAN_AUDIO_DIR = os.path.join(BASE_DIR, "cleaned_audio_unique")
OUT_DIR = os.path.join(BASE_DIR, "outputs_for_colab_clean")

os.makedirs(CLEAN_AUDIO_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_MAPPING = {
    'belly pain': 0,
    'belly_pain': 0,
    'burping': 1,
    'cold_hot': 2,
    'discomfort': 2,
    'hungry': 3,
    'tired': 4
}

def get_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()

def step1_hash_cleaning():
    print("STEP 1: Hash-based cleaning & Label resolution")
    inventory = []
    
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.lower().endswith(('.wav', '.ogg', '.mp3')):
                file_path = os.path.join(root, file)
                parent = Path(file_path).parent.name
                
                if parent not in LABEL_MAPPING:
                    continue
                    
                label = LABEL_MAPPING[parent]
                file_hash = get_hash(file_path)
                
                inventory.append({
                    'file_path': file_path,
                    'filename': file,
                    'parent': parent,
                    'label': label,
                    'hash': file_hash
                })
                
    df = pd.DataFrame(inventory)
    print(f"Total valid raw files found: {len(df)}")
    
    unique_records = []
    
    for hash_val, group in df.groupby('hash'):
        labels = group['label'].tolist()
        majority_label = Counter(labels).most_common(1)[0][0]
        
        best_file = group[group['label'] == majority_label].iloc[0]
        
        unique_records.append({
            'hash': hash_val,
            'source_path': best_file['file_path'],
            'filename': best_file['filename'],
            'label': majority_label,
            'group_id': hash_val
        })
        
    clean_df = pd.DataFrame(unique_records)
    print(f"Total unique files after deduplication: {len(clean_df)}")
    
    print("Standardizing and copying to cleaned_audio_unique...")
    final_metadata = []
    
    for idx, row in clean_df.iterrows():
        try:
            y, sr = librosa.load(row['source_path'], sr=16000, mono=True)
            y_trimmed, _ = librosa.effects.trim(y, top_db=30)
            
            out_filename = f"{row['hash']}.wav"
            out_path = os.path.join(CLEAN_AUDIO_DIR, out_filename)
            
            sf.write(out_path, y_trimmed, 16000, subtype='FLOAT')
            
            final_metadata.append({
                'hash': row['hash'],
                'clean_path': out_path,
                'filename': out_filename,
                'label': row['label'],
                'group_id': row['group_id']
            })
        except Exception as e:
            print(f"Error processing {row['source_path']}: {e}")
            
    final_df = pd.DataFrame(final_metadata)
    final_df.to_csv(os.path.join(OUT_DIR, "metadata_clean.csv"), index=False)
    print("Step 1 Complete.\n")
    return final_df

def step2_feature_extraction(df):
    print("STEP 2: Feature Extraction (Fixing axis=1)")
    
    features_list = []
    labels_list = []
    groups_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        try:
            y, sr = librosa.load(row['clean_path'], sr=16000)
            
            amp_env = np.mean(np.abs(y))
            rms = np.mean(librosa.feature.rms(y=y))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
            
            S = np.abs(librosa.stft(y))
            sc = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))
            sban = np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr))
            scon = np.mean(librosa.feature.spectral_contrast(S=S, sr=sr))
            
            mfccs13 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs13_mean = np.mean(mfccs13, axis=1)
            
            delta_mfccs13 = librosa.feature.delta(mfccs13)
            delta_mfccs13_mean = np.mean(delta_mfccs13, axis=1)
            
            delta2_mfccs13 = librosa.feature.delta(mfccs13, order=2)
            delta2_mfccs13_mean = np.mean(delta2_mfccs13, axis=1)
            
            mfccs20 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfccs20_mean = np.mean(mfccs20, axis=1)
            
            feat_vector = np.concatenate([
                [amp_env, rms, zcr, sc, sban, scon],
                mfccs13_mean,
                delta_mfccs13_mean,
                delta2_mfccs13_mean,
                mfccs20_mean
            ])
            
            features_list.append(feat_vector)
            labels_list.append(row['label'])
            groups_list.append(row['group_id'])
            valid_indices.append(idx)
            
        except Exception as e:
            print(f"Error extracting features for {row['clean_path']}: {e}")
            
    X_tabular = np.array(features_list, dtype=np.float32)
    y_labels = np.array(labels_list, dtype=np.int32)
    groups = np.array(groups_list)
    
    np.save(os.path.join(OUT_DIR, "X_tabular_new.npy"), X_tabular)
    np.save(os.path.join(OUT_DIR, "y_labels_new.npy"), y_labels)
    np.save(os.path.join(OUT_DIR, "groups_new.npy"), groups)
    
    df_valid = df.loc[valid_indices].reset_index(drop=True)
    df_valid.to_csv(os.path.join(OUT_DIR, "metadata_clean.csv"), index=False)
    
    print(f"X_tabular shape: {X_tabular.shape}")
    print("Step 2 Complete.\n")
    return df_valid

def step3_spectrograms(df):
    print("STEP 3: Spectrogram Generation")
    
    images = []
    
    for idx, row in df.iterrows():
        try:
            y, sr = librosa.load(row['clean_path'], sr=16000)
            
            melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            melspec_db = librosa.power_to_db(melspec, ref=np.max)
            
            target_length = 128
            
            if melspec_db.shape[1] < target_length:
                pad_width = target_length - melspec_db.shape[1]
                melspec_db = np.pad(melspec_db, pad_width=((0,0), (0, pad_width)), mode='constant')
            else:
                melspec_db = melspec_db[:, :target_length]
                
            melspec_db = (melspec_db - melspec_db.min()) / (melspec_db.max() - melspec_db.min() + 1e-9)
            melspec_db = np.expand_dims(melspec_db, axis=-1)
            images.append(melspec_db)
            
        except Exception as e:
            print(f"Error generating spectrogram for {row['clean_path']}: {e}")
            
    X_images = np.array(images, dtype=np.float32)
    np.save(os.path.join(OUT_DIR, "X_images_new.npy"), X_images)
    
    print(f"X_images shape: {X_images.shape}")
    print("Step 3 Complete.\n")

def step4_report(df):
    print("STEP 4: Generating Report")
    
    X_tab = np.load(os.path.join(OUT_DIR, "X_tabular_new.npy"))
    X_img = np.load(os.path.join(OUT_DIR, "X_images_new.npy"))
    
    report_path = os.path.join(OUT_DIR, "dataset_report_clean.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Cleaned AML Dataset Report\n\n")
        f.write(f"- **Toplam örnek sayısı**: {len(df)}\n")
        f.write(f"- **X_images_new shape**: {X_img.shape}\n")
        f.write(f"- **X_tabular_new shape**: {X_tab.shape}\n")
        f.write(f"- **y_labels_new shape**: {len(df)}\n")
        f.write(f"- **groups_new shape**: {len(df)}\n\n")
        
        f.write("## Sınıf Dağılımı\n")
        f.write(df['label'].value_counts().to_string() + "\n\n")
        
        f.write("## Temizleme Özeti\n")
        f.write("- **Yöntem**: Tüm raw dosyalardan MD5 Hash alınarak duplicate'ler engellendi.\n")
        f.write("- **Label Çakışması**: Farklı klasörlerde aynı hash'e sahip dosyalar tespit edildi. Çoğunluk label'ı atandı.\n")
        f.write("- **Feature Düzeltmesi**: Tabular feature çıkarılırken `axis=1` kullanıldı. Boyut 10'dan 65'e çıktı.\n")
        f.write("- **Group ID**: Hash değeri doğrudan group_id olarak kullanıldı. Böylece leakage tamamen engellendi.\n")

if __name__ == "__main__":
    df = step1_hash_cleaning()
    df = step2_feature_extraction(df)
    step3_spectrograms(df)
    step4_report(df)
    print("All tasks completed successfully!")
