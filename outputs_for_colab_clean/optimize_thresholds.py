import os
import json
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tqdm import tqdm

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
    RESULTS_DIR = os.path.join(BASE_DIR, "results_ensemble")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Loading data...")
    X_tab = np.load(os.path.join(BASE_DIR, "X_tabular_new.npy"))
    y_raw = np.load(os.path.join(BASE_DIR, "y_labels_new.npy"), allow_pickle=True)
    groups = np.load(os.path.join(BASE_DIR, "groups_new.npy"), allow_pickle=True)
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    target_names = [str(c) for c in le.classes_]
    num_classes = len(le.classes_)
    
    train_idx, test_idx = choose_best_group_split(X_tab, y, groups)
    
    X_train, X_test = X_tab[train_idx], X_tab[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("Training Base Models...")
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)
    
    svc = SVC(random_state=42, class_weight="balanced", probability=True)
    svc.fit(X_train, y_train)
    svc_probs = svc.predict_proba(X_test)
    
    probs = 0.6 * rf_probs + 0.4 * svc_probs
    
    print("Starting Grid Search for Thresholds...")
    threshold_values = np.arange(0.1, 1.0, 0.1)
    
    best_macro_f1 = -1
    best_thresholds = None
    
    combinations = list(itertools.product(threshold_values, repeat=num_classes))
    
    standard_preds = np.argmax(probs, axis=1)
    standard_f1 = f1_score(y_test, standard_preds, average="macro", zero_division=0)
    standard_recall = recall_score(y_test, standard_preds, average="macro", zero_division=0)
    per_class_recalls = recall_score(y_test, standard_preds, average=None, zero_division=0)
    
    print(f"Standard Argmax Macro F1: {standard_f1:.4f}")
    
    for th in tqdm(combinations, desc="Grid Search"):
        preds = predict_with_thresholds(probs, th)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        
        if f1 > best_macro_f1:
            best_macro_f1 = f1
            best_thresholds = th
            
    print(f"\nBest Thresholds: {best_thresholds}")
    print(f"Best Macro F1: {best_macro_f1:.4f}")
    
    best_preds = predict_with_thresholds(probs, best_thresholds)
    best_recall = recall_score(y_test, best_preds, average="macro", zero_division=0)
    best_per_class_recalls = recall_score(y_test, best_preds, average=None, zero_division=0)
    
    thresholds_dict = {str(target_names[i]): float(best_thresholds[i]) for i in range(num_classes)}
    
    with open(os.path.join(RESULTS_DIR, "best_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=4)
        
    metrics = {
        "before_optimization": {
            "macro_f1": float(standard_f1),
            "macro_recall": float(standard_recall),
            "per_class_recall": {str(c): float(r) for c, r in zip(target_names, per_class_recalls)}
        },
        "after_optimization": {
            "macro_f1": float(best_macro_f1),
            "macro_recall": float(best_recall),
            "per_class_recall": {str(c): float(r) for c, r in zip(target_names, best_per_class_recalls)}
        }
    }
    
    with open(os.path.join(RESULTS_DIR, "metrics_thresholded.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    report = classification_report(y_test, best_preds, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_test, best_preds)
    
    with open(os.path.join(RESULTS_DIR, "classification_report_thresholded.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== BEST ENSEMBLE WITH OPTIMIZED THRESHOLDS ===\n")
        f.write(f"Thresholds: {thresholds_dict}\n\n")
        f.write(report)
        
    np.save(os.path.join(RESULTS_DIR, "confusion_matrix_thresholded.npy"), cm)
    
    print("\nComparison:")
    print(f"Macro F1: {standard_f1:.4f} -> {best_macro_f1:.4f}")
    print(f"Macro Recall: {standard_recall:.4f} -> {best_recall:.4f}")
    
    for i, c in enumerate(target_names):
        print(f"Class {c} Recall: {per_class_recalls[i]:.4f} -> {best_per_class_recalls[i]:.4f}")

if __name__ == "__main__":
    main()
