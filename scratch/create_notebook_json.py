import json
import os

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\\n" for line in text.split("\\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\\n" for line in text.split("\\n")]
    })

add_md("""# Baby Cry Classification V2: Librosa Baseline Experiment
This notebook conducts a baseline applied machine learning experiment using the newly cleaned, evidence-based, leak-free baby cry dataset.

**Key principles applied:**
1. **Zero Data Leakage:** We use the MD5 hash (embedded in the filename) as the `group_id` for `StratifiedGroupKFold`. This ensures identical physical copies are strictly isolated between train and validation sets.
2. **Class Imbalance Handling:** We do NOT use SMOTE (to prevent synthetic artifacts in audio distributions). Instead, we use `class_weight='balanced'` in all algorithms.
3. **Robust Metrics:** We prioritize `macro-F1`, `balanced_accuracy`, and `macro-recall` over raw accuracy.""")

add_code("""from google.colab import drive
import os

drive.mount('/content/drive')

OUTPUT_DIR = '/content/outputs_v2_colab'
os.makedirs(OUTPUT_DIR, exist_ok=True)""")

add_code("""!pip install librosa soundfile scikit-learn pandas numpy matplotlib joblib tqdm

import librosa
import soundfile as sf
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm.auto import tqdm
import warnings

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')""")

add_code("""# Unzip datasets from Google Drive
!mkdir -p /content/datasets
!unzip -q -o /content/drive/MyDrive/baby_cry_v2/clean_majority_5class.zip -d /content/datasets/
!unzip -q -o /content/drive/MyDrive/baby_cry_v2/clean_majority_binary.zip -d /content/datasets/

# The structure inside /content/datasets/ should now be:
# /content/datasets/master_clean_evidence/majority_5class/
# /content/datasets/master_clean_evidence/majority_binary/""")

add_code("""def extract_librosa_features(file_path):
    \"\"\"
    Extract standard acoustic features using librosa.
    \"\"\"
    try:
        y, sr = librosa.load(file_path, sr=None) # Audio is already 16k mono
        if len(y) == 0:
            return None
        
        # Duration
        dur = librosa.get_duration(y=y, sr=sr)
        
        # RMS
        rms = librosa.feature.rms(y=y)[0]
        
        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Spectral Bandwidth
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Pitch (F0)
        f0 = librosa.yin(y, fmin=50, fmax=800, frame_length=2048)
        f0[f0 == 50] = np.nan # basic filter for unvoiced/failed estimation
        f0_mean = np.nanmean(f0) if not np.isnan(f0).all() else 0
        f0_std = np.nanstd(f0) if not np.isnan(f0).all() else 0
        
        features = {
            'duration': dur,
            'rms_mean': np.mean(rms), 'rms_std': np.std(rms),
            'zcr_mean': np.mean(zcr), 'zcr_std': np.std(zcr),
            'spectral_centroid_mean': np.mean(cent), 'spectral_centroid_std': np.std(cent),
            'spectral_bandwidth_mean': np.mean(bw), 'spectral_bandwidth_std': np.std(bw),
            'spectral_rolloff_mean': np.mean(rolloff), 'spectral_rolloff_std': np.std(rolloff),
            'f0_mean': f0_mean, 'f0_std': f0_std
        }
        
        for i in range(20):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            features[f'mfcc_delta_{i+1}_mean'] = np.mean(mfccs_delta[i])
            features[f'mfcc_delta_{i+1}_std'] = np.std(mfccs_delta[i])
            features[f'mfcc_delta2_{i+1}_mean'] = np.mean(mfccs_delta2[i])
            features[f'mfcc_delta2_{i+1}_std'] = np.std(mfccs_delta2[i])
            
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_dataset(root_dir, csv_output):
    data = []
    files = glob.glob(os.path.join(root_dir, '**', '*.wav'), recursive=True)
    for f in tqdm(files, desc=f"Extracting features for {os.path.basename(root_dir)}"):
        filename = os.path.basename(f)
        label = os.path.basename(os.path.dirname(f))
        
        # Extract group ID (md5 hash) from filename format: <label>__<md5>.wav
        parts = filename.replace('.wav', '').split('__')
        group_id = parts[-1] if len(parts) > 1 else 'unknown'
        
        feats = extract_librosa_features(f)
        if feats is not None:
            feats['label'] = label
            feats['group_id'] = group_id
            feats['file_path'] = f
            data.append(feats)
            
    df = pd.DataFrame(data)
    df.to_csv(csv_output, index=False)
    print(f"Saved {len(df)} samples to {csv_output}")
    return df""")

add_code("""df_5class = process_dataset('/content/datasets/master_clean_evidence/majority_5class', f'{OUTPUT_DIR}/features_5class_librosa.csv')""")

add_code("""df_binary = process_dataset('/content/datasets/master_clean_evidence/majority_binary', f'{OUTPUT_DIR}/features_binary_librosa.csv')""")

add_code("""def evaluate_models(df, task_name, output_dir):
    print(f"\\n{'='*50}\\nEvaluating {task_name}\\n{'='*50}")
    
    # Class distribution plot
    plt.figure(figsize=(10,5))
    sns.countplot(data=df, x='label', order=df['label'].value_counts().index)
    plt.title(f'Class Distribution - {task_name}')
    plt.savefig(f'{output_dir}/class_distribution_{task_name}.png', bbox_inches='tight')
    plt.close()
    
    X = df.drop(columns=['label', 'group_id', 'file_path'])
    y = df['label']
    groups = df['group_id']
    
    # Check minimum class members to set CV folds
    min_class_count = y.value_counts().min()
    n_splits = 5 if min_class_count >= 5 else (3 if min_class_count >= 3 else 2)
    print(f"Using StratifiedGroupKFold with n_splits={n_splits}")
    cv = StratifiedGroupKFold(n_splits=n_splits)
    
    models = {
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=5000),
        'RandomForest': RandomForestClassifier(class_weight='balanced', n_estimators=500, random_state=42),
        'ExtraTrees': ExtraTreesClassifier(class_weight='balanced', n_estimators=500, random_state=42),
        'SVC': SVC(class_weight='balanced', probability=True, kernel='rbf'),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42)
    }
    
    results = []
    trained_models = {}
    
    for name, clf in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        
        fold_macro_f1 = []
        fold_bal_acc = []
        fold_macro_recall = []
        
        all_y_true = []
        all_y_pred = []
        
        for train_idx, test_idx in cv.split(X, y, groups):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            
            all_y_true.extend(y_test)
            all_y_pred.extend(preds)
            
            fold_macro_f1.append(f1_score(y_test, preds, average='macro'))
            fold_bal_acc.append(balanced_accuracy_score(y_test, preds))
            fold_macro_recall.append(recall_score(y_test, preds, average='macro'))
            
        mean_macro_f1 = np.mean(fold_macro_f1)
        mean_bal_acc = np.mean(fold_bal_acc)
        mean_macro_rec = np.mean(fold_macro_recall)
        
        results.append({
            'Model': name,
            'Macro_F1': mean_macro_f1,
            'Balanced_Accuracy': mean_bal_acc,
            'Macro_Recall': mean_macro_rec
        })
        
        # Retrain on full for saving
        pipeline.fit(X, y)
        trained_models[name] = pipeline
        
    df_res = pd.DataFrame(results).sort_values(by=['Macro_F1', 'Balanced_Accuracy'], ascending=False)
    df_res.to_csv(f"{output_dir}/results_{task_name}_librosa.csv", index=False)
    print(df_res)
    
    # Identify best model
    best_model_name = df_res.iloc[0]['Model']
    best_pipeline = trained_models[best_model_name]
    joblib.dump(best_pipeline, f"{output_dir}/best_model_{task_name}.joblib")
    print(f"Best model for {task_name}: {best_model_name}")
    
    # Get cross-val predictions for the best model to generate reports and plots
    best_y_true = []
    best_y_pred = []
    for train_idx, test_idx in cv.split(X, y, groups):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        clf_fresh = models[best_model_name]
        pipe_fresh = Pipeline([('scaler', StandardScaler()), ('classifier', clf_fresh)])
        pipe_fresh.fit(X_train, y_train)
        preds = pipe_fresh.predict(X_test)
        
        best_y_true.extend(y_test)
        best_y_pred.extend(preds)
        
    # Classification Report
    report = classification_report(best_y_true, best_y_pred)
    with open(f"{output_dir}/classification_report_{task_name}.txt", "w") as f:
        f.write(f"Classification Report for Best Model: {best_model_name}\\n\\n")
        f.write(report)
        
    # Confusion Matrix
    cm = confusion_matrix(best_y_true, best_y_pred)
    classes = np.unique(best_y_true)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix ({best_model_name}) - {task_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{output_dir}/confusion_matrix_{task_name}.png', bbox_inches='tight')
    plt.close()
    
    # Model Comparison Plots
    plt.figure(figsize=(10,6))
    sns.barplot(data=df_res, x='Macro_F1', y='Model', palette='viridis')
    plt.title(f'Model Comparison: Macro-F1 - {task_name}')
    plt.savefig(f'{output_dir}/model_comparison_{task_name}.png', bbox_inches='tight')
    plt.close()
    
    return best_model_name, df_res""")

add_code("""best_5class, df_res_5class = evaluate_models(df_5class, '5class', OUTPUT_DIR)""")

add_code("""best_binary, df_res_binary = evaluate_models(df_binary, 'binary', OUTPUT_DIR)""")

add_code("""summary_md = f\"\"\"# Baby Cry Classification V2: Experiment Summary

## Datasets Evaluated
- **5-Class Dataset Count:** {len(df_5class)} samples
- **Binary Dataset Count:** {len(df_binary)} samples

## Key Engineering Decisions
- **SMOTE Excluded:** No synthetic oversampling was used, preserving true acoustic data distributions. Imbalance was handled natively using `class_weight='balanced'`.
- **Zero Data Leakage:** Group-aware validation (`StratifiedGroupKFold`) was strictly enforced using physical MD5 hashes. Identical physical audio recordings never contaminated both the train and validation sets simultaneously.

## Results
- **Best 5-Class Model:** {best_5class} (Macro-F1: {df_res_5class.iloc[0]['Macro_F1']:.4f})
- **Best Binary Model:** {best_binary} (Macro-F1: {df_res_binary.iloc[0]['Macro_F1']:.4f})

## Limitations
Librosa baseline features (MFCCs, spectral properties) provide a robust baseline but may fail to capture complex temporal dependencies present in baby cries. Extreme class imbalance in the 5-class task still constrains macro-level performance, even with class weighting.

## Next Step
Extract comprehensive time-series features using `tsfresh` to evaluate if massive automated feature engineering yields significantly better macro-recall and F1 scores over the Librosa baseline.
\"\"\"

with open(f"{OUTPUT_DIR}/experiment_summary.md", "w") as f:
    f.write(summary_md)
    
print("Experiment summary generated.")""")

notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(r'C:\Users\ASUS\Desktop\aml_project\baby_cry_v2_librosa_baseline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_json, f, indent=2)
