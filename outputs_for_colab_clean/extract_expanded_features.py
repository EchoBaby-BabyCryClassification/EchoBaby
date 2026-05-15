import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import json
import warnings
warnings.filterwarnings('ignore')

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        
        # Duration
        duration = librosa.get_duration(y=y, sr=sr)
        
        # RMS / Energy
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # Spectral features
        S = np.abs(librosa.stft(y))
        sc = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        sc_mean = np.mean(sc)
        sc_std = np.std(sc)
        
        sban = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
        sban_mean = np.mean(sban)
        sban_std = np.std(sban)
        
        sroll = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
        sroll_mean = np.mean(sroll)
        sroll_std = np.std(sroll)
        
        # Pitch (f0)
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        
        # Handle nan for unvoiced frames
        if f0 is not None and len(f0) > 0:
            pitch_mean = np.mean(f0)
            pitch_std = np.std(f0)
            pitch_min = np.min(f0)
            pitch_max = np.max(f0)
        else:
            pitch_mean = 0.0
            pitch_std = 0.0
            pitch_min = 0.0
            pitch_max = 0.0
            
        # MFCCs
        mfccs13 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs13_mean = np.mean(mfccs13, axis=1)
        
        delta_mfccs13 = librosa.feature.delta(mfccs13)
        delta_mfccs13_mean = np.mean(delta_mfccs13, axis=1)
        
        delta2_mfccs13 = librosa.feature.delta(mfccs13, order=2)
        delta2_mfccs13_mean = np.mean(delta2_mfccs13, axis=1)
        
        mfccs20 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs20_mean = np.mean(mfccs20, axis=1)
        
        features = np.concatenate([
            [duration, rms_mean, rms_std, zcr_mean, zcr_std, sc_mean, sc_std, 
             sban_mean, sban_std, sroll_mean, sroll_std, pitch_mean, pitch_std, pitch_min, pitch_max],
            mfccs13_mean,
            delta_mfccs13_mean,
            delta2_mfccs13_mean,
            mfccs20_mean
        ])
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    BASE_DIR = r"C:\Users\ASUS\Desktop\aml_project\outputs_for_colab_clean"
    RESULTS_DIR = os.path.join(BASE_DIR, "results_tabular_cv_v2")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    csv_path = os.path.join(BASE_DIR, "metadata_clean.csv")
    df = pd.read_csv(csv_path)
    
    print("Extracting Expanded Features (v2)...")
    
    features_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        file_path = row['clean_path']
        f = extract_features(file_path)
        features_list.append(f)
        
    X_tab_v2 = np.array(features_list, dtype=np.float32)
    np.save(os.path.join(BASE_DIR, "X_tabular_v2.npy"), X_tab_v2)
    print(f"X_tabular_v2 saved with shape: {X_tab_v2.shape}")
    
    print("Loading labels and groups...")
    y_raw = np.load(os.path.join(BASE_DIR, "y_labels_new.npy"), allow_pickle=True)
    groups = np.load(os.path.join(BASE_DIR, "groups_new.npy"), allow_pickle=True)
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    target_names = [str(c) for c in le.classes_]
    
    models = {
        "RandomForest": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "ExtraTrees": ExtraTreesClassifier(random_state=42, class_weight="balanced"),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42, class_weight="balanced"),
        "SVC": SVC(random_state=42, class_weight="balanced"),
        "LogisticRegression": LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
    }
    
    results = {name: {"macro_f1": [], "macro_recall": []} for name in models}
    for name in models:
        for c in target_names:
            results[name][f"recall_class_{c}"] = []
            
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\nEvaluating models on V2 features with 5-Fold StratifiedGroupKFold...")
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        fold_f1s = []
        fold_recalls = []
        fold_per_class_recalls = {c: [] for c in target_names}
        
        try:
            for train_idx, test_idx in sgkf.split(X_tab_v2, y, groups):
                X_train, X_test = X_tab_v2[train_idx], X_tab_v2[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                fold_f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
                fold_recalls.append(recall_score(y_test, y_pred, average="macro", zero_division=0))
                
                recalls = recall_score(y_test, y_pred, average=None, labels=np.arange(len(target_names)), zero_division=0)
                for i, c in enumerate(target_names):
                    fold_per_class_recalls[c].append(recalls[i])
                    
            results[model_name]["macro_f1"] = fold_f1s
            results[model_name]["macro_recall"] = fold_recalls
            for c in target_names:
                results[model_name][f"recall_class_{c}"] = fold_per_class_recalls[c]
                
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            
    summary = {}
    best_overall_f1 = -1
    best_model_name = None
    
    for name in results:
        if len(results[name]["macro_f1"]) > 0:
            mean_f1 = float(np.mean(results[name]["macro_f1"]))
            if mean_f1 > best_overall_f1:
                best_overall_f1 = mean_f1
                best_model_name = name
                
            summary[name] = {
                "macro_f1_mean": mean_f1,
                "macro_f1_std": float(np.std(results[name]["macro_f1"])),
                "macro_recall_mean": float(np.mean(results[name]["macro_recall"])),
                "macro_recall_std": float(np.std(results[name]["macro_recall"]))
            }
            for c in target_names:
                summary[name][f"recall_class_{c}_mean"] = float(np.mean(results[name][f"recall_class_{c}"]))
            
    with open(os.path.join(RESULTS_DIR, "cv_results_v2.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\nBest Model on V2: {best_model_name} with Mean Macro F1: {best_overall_f1:.4f}")
    print("Done! Results saved to outputs_for_colab_clean/results_tabular_cv_v2")

if __name__ == "__main__":
    main()
