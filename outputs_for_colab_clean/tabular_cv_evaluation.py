import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def main():
    BASE_DIR = r"C:\Users\ASUS\Desktop\aml_project\outputs_for_colab_clean"
    RESULTS_DIR = os.path.join(BASE_DIR, "results_tabular_cv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Loading data...")
    X_tab = np.load(os.path.join(BASE_DIR, "X_tabular_new.npy"))
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
    
    best_overall_f1 = -1
    best_model_name = None
    best_model_fold_preds = None
    best_model_fold_y = None
    
    print("\nEvaluating models with 5-Fold StratifiedGroupKFold...")
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        fold_f1s = []
        fold_recalls = []
        fold_per_class_recalls = {c: [] for c in target_names}
        
        all_y_true = []
        all_y_pred = []
        
        try:
            for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_tab, y, groups)):
                X_train, X_test = X_tab[train_idx], X_tab[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)
                
                fold_f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
                fold_recalls.append(recall_score(y_test, y_pred, average="macro", zero_division=0))
                
                recalls = recall_score(y_test, y_pred, average=None, labels=np.arange(len(target_names)), zero_division=0)
                for i, c in enumerate(target_names):
                    fold_per_class_recalls[c].append(recalls[i])
                    
            results[model_name]["macro_f1"] = fold_f1s
            results[model_name]["macro_recall"] = fold_recalls
            for c in target_names:
                results[model_name][f"recall_class_{c}"] = fold_per_class_recalls[c]
                
            mean_f1 = np.mean(fold_f1s)
            if mean_f1 > best_overall_f1:
                best_overall_f1 = mean_f1
                best_model_name = model_name
                best_model_fold_preds = all_y_pred
                best_model_fold_y = all_y_true
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            
    summary = {}
    for name in results:
        if len(results[name]["macro_f1"]) > 0:
            summary[name] = {
                "macro_f1_mean": float(np.mean(results[name]["macro_f1"])),
                "macro_f1_std": float(np.std(results[name]["macro_f1"])),
                "macro_recall_mean": float(np.mean(results[name]["macro_recall"])),
                "macro_recall_std": float(np.std(results[name]["macro_recall"]))
            }
            for c in target_names:
                summary[name][f"recall_class_{c}_mean"] = float(np.mean(results[name][f"recall_class_{c}"]))
            
    with open(os.path.join(RESULTS_DIR, "cv_results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\nBest Model: {best_model_name} with Mean Macro F1: {best_overall_f1:.4f}")
    
    if best_model_name:
        report = classification_report(best_model_fold_y, best_model_fold_preds, target_names=target_names, zero_division=0)
        cm = confusion_matrix(best_model_fold_y, best_model_fold_preds)
        
        with open(os.path.join(RESULTS_DIR, "best_model_classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(f"=== BEST MODEL (Cross-Validated Aggregated): {best_model_name} ===\n\n")
            f.write(report)
            
        np.save(os.path.join(RESULTS_DIR, "best_model_confusion_matrix.npy"), cm)
    print("Results written to outputs_for_colab_clean/results_tabular_cv")

if __name__ == "__main__":
    main()
