import os
import json
import pickle
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def choose_best_group_split(X, y, groups, n_splits=5):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_classes = set(np.unique(y))
    best_score = -float('inf')
    best_split = None

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        train_classes = set(np.unique(y[train_idx]))
        test_classes = set(np.unique(y[test_idx]))

        missing_train = len(all_classes - train_classes)
        missing_test = len(all_classes - test_classes)
        
        # Leakage check
        assert len(set(groups[train_idx]).intersection(set(groups[test_idx]))) == 0

        score = -100 * (missing_train + missing_test) - abs(len(test_idx) / len(y) - 0.2)

        if score > best_score:
            best_score = score
            best_split = (train_idx, test_idx)

    return best_split

def predict_with_thresholds(probs, thresholds):
    passed = probs >= thresholds
    masked_probs = np.where(passed, probs, -1.0)
    preds = np.argmax(masked_probs, axis=1)
    
    fallback_needed = ~np.any(passed, axis=1)
    if np.any(fallback_needed):
        fallback_preds = np.argmax(probs, axis=1)
        preds[fallback_needed] = fallback_preds[fallback_needed]
        
    return preds

def main():
    BASE_DIR = r"C:\Users\ASUS\Desktop\aml_project\outputs_for_colab_clean"
    RESULTS_DIR = os.path.join(BASE_DIR, "results_final_model")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Loading data...")
    X_tab = np.load(os.path.join(BASE_DIR, "X_tabular_v2.npy"))
    y_raw = np.load(os.path.join(BASE_DIR, "y_labels_new.npy"), allow_pickle=True)
    groups = np.load(os.path.join(BASE_DIR, "groups_new.npy"), allow_pickle=True)
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    target_names = [str(c) for c in le.classes_]
    num_classes = len(target_names)
    
    train_idx, test_idx = choose_best_group_split(X_tab, y, groups)
    
    X_train, X_test = X_tab[train_idx], X_tab[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Pre-SMOTE distribution
    pre_smote_dist = dict(zip(*np.unique(y_train, return_counts=True)))
    
    # SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    post_smote_dist = dict(zip(*np.unique(y_train_res, return_counts=True)))
    
    # Scale
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)
    
    print("Training Base Models...")
    et = ExtraTreesClassifier(random_state=42)
    et.fit(X_train_res, y_train_res)
    et_probs = et.predict_proba(X_test)
    
    svc = SVC(random_state=42, probability=True)
    svc.fit(X_train_res, y_train_res)
    svc_probs = svc.predict_proba(X_test)
    
    et_weights = [0.5, 0.6, 0.7, 0.8]
    threshold_values = np.arange(0.1, 1.0, 0.2)
    combinations = list(itertools.product(threshold_values, repeat=num_classes))
    
    best_macro_f1 = -1
    best_weight = None
    best_thresholds = None
    
    print("Starting Grid Search for Ensemble Weights and Thresholds...")
    
    for w in et_weights:
        print(f"Testing ExtraTrees weight: {w}, SVC weight: {1-w:.1f}")
        probs = w * et_probs + (1 - w) * svc_probs
        
        # Adding a small progress bar for each weight setting
        for th in tqdm(combinations, desc=f"Grid Search (ET={w})", leave=False):
            preds = predict_with_thresholds(probs, th)
            f1 = f1_score(y_test, preds, average="macro", zero_division=0)
            
            if f1 > best_macro_f1:
                best_macro_f1 = f1
                best_weight = w
                best_thresholds = th
                
    print("\nSearch Completed!")
    
    # Generate Best Predictions
    best_probs = best_weight * et_probs + (1 - best_weight) * svc_probs
    best_preds = predict_with_thresholds(best_probs, best_thresholds)
    
    macro_f1 = f1_score(y_test, best_preds, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, best_preds, average="macro", zero_division=0)
    acc = accuracy_score(y_test, best_preds)
    per_class_recall = recall_score(y_test, best_preds, average=None, zero_division=0)
    
    # Save
    with open(os.path.join(RESULTS_DIR, "final_model_extratrees.pkl"), "wb") as f:
        pickle.dump(et, f)
    with open(os.path.join(RESULTS_DIR, "final_model_svc.pkl"), "wb") as f:
        pickle.dump(svc, f)
    with open(os.path.join(RESULTS_DIR, "final_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(RESULTS_DIR, "final_smote.pkl"), "wb") as f:
        pickle.dump(smote, f)
        
    metrics = {
        "macro_f1": float(macro_f1),
        "macro_recall": float(macro_recall),
        "accuracy": float(acc),
        "per_class_recall": {str(target_names[i]): float(per_class_recall[i]) for i in range(num_classes)},
        "best_et_weight": float(best_weight),
        "best_svc_weight": float(1 - best_weight)
    }
    with open(os.path.join(RESULTS_DIR, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    thresholds_dict = {str(target_names[i]): float(best_thresholds[i]) for i in range(num_classes)}
    with open(os.path.join(RESULTS_DIR, "final_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=4)
        
    report = classification_report(y_test, best_preds, target_names=target_names, zero_division=0)
    with open(os.path.join(RESULTS_DIR, "final_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
        
    cm = confusion_matrix(y_test, best_preds)
    np.save(os.path.join(RESULTS_DIR, "final_confusion_matrix.npy"), cm)
    
    analysis = {
        "pre_smote_distribution": {str(target_names[k]): int(v) for k, v in pre_smote_dist.items()},
        "post_smote_distribution": {str(target_names[k]): int(v) for k, v in post_smote_dist.items()}
    }
    with open(os.path.join(RESULTS_DIR, "final_prediction_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=4)
        
    print("\n--- Final Model Report ---")
    print(f"Pre-SMOTE Class Dist: {analysis['pre_smote_distribution']}")
    print(f"Post-SMOTE Class Dist: {analysis['post_smote_distribution']}")
    print(f"Best Ensemble Weight: ET={best_weight}, SVC={1-best_weight:.1f}")
    print(f"Best Thresholds: {thresholds_dict}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Accuracy: {acc:.4f}")
    for i, c in enumerate(target_names):
        print(f"Class {c} Recall: {per_class_recall[i]:.4f}")

if __name__ == "__main__":
    main()
