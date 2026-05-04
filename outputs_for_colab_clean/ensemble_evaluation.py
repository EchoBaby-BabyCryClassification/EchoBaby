import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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
    
    train_idx, test_idx = choose_best_group_split(X_tab, y, groups)
    
    X_train, X_test = X_tab[train_idx], X_tab[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("Training RandomForest...")
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)
    rf_preds = rf.predict(X_test)
    
    print("Training SVC...")
    svc = SVC(random_state=42, class_weight="balanced", probability=True)
    svc.fit(X_train, y_train)
    svc_probs = svc.predict_proba(X_test)
    svc_preds = svc.predict(X_test)
    
    print("Evaluating Ensembles...")
    
    # A) Soft Voting (0.6 RF + 0.4 SVC)
    prob_A = 0.6 * rf_probs + 0.4 * svc_probs
    pred_A = np.argmax(prob_A, axis=1)
    
    # B) Balanced Voting (0.5 RF + 0.5 SVC)
    prob_B = 0.5 * rf_probs + 0.5 * svc_probs
    pred_B = np.argmax(prob_B, axis=1)
    
    # C) Class-weighted fusion
    prob_C = np.zeros_like(rf_probs)
    hungry_idx = np.where(le.classes_ == 3)[0][0]
    
    for i in range(len(le.classes_)):
        if i == hungry_idx:
            prob_C[:, i] = 0.8 * rf_probs[:, i] + 0.2 * svc_probs[:, i]
        else:
            prob_C[:, i] = 0.2 * rf_probs[:, i] + 0.8 * svc_probs[:, i]
            
    prob_C = prob_C / np.sum(prob_C, axis=1, keepdims=True)
    pred_C = np.argmax(prob_C, axis=1)
    
    def evaluate(preds):
        return {
            "macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_test, preds, average="macro", zero_division=0)),
            "per_class_recall": {str(c): float(r) for c, r in zip(le.classes_, recall_score(y_test, preds, average=None, zero_division=0))}
        }
        
    results = {
        "A_SoftVoting_60_40": evaluate(pred_A),
        "B_BalancedVoting_50_50": evaluate(pred_B),
        "C_ClassWeightedFusion": evaluate(pred_C),
        "RF_Baseline": evaluate(rf_preds),
        "SVC_Baseline": evaluate(svc_preds)
    }
    
    best_method = max(["A_SoftVoting_60_40", "B_BalancedVoting_50_50", "C_ClassWeightedFusion"], key=lambda k: results[k]["macro_f1"])
    best_preds = pred_A if best_method == "A_SoftVoting_60_40" else (pred_B if best_method == "B_BalancedVoting_50_50" else pred_C)
    
    print(f"Best Method: {best_method} with Macro F1: {results[best_method]['macro_f1']:.4f}")
    
    with open(os.path.join(RESULTS_DIR, "ensemble_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    report = classification_report(y_test, best_preds, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_test, best_preds)
    
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== BEST ENSEMBLE MODEL: {best_method} ===\n\n")
        f.write(report)
        
    np.save(os.path.join(RESULTS_DIR, "confusion_matrix.npy"), cm)
    
    rf_wrong_svc_right = []
    svc_wrong_rf_right = []
    
    for i in range(len(y_test)):
        true_label = int(y_test[i])
        rf_p = int(rf_preds[i])
        svc_p = int(svc_preds[i])
        
        orig_idx = int(test_idx[i])
        
        if rf_p != true_label and svc_p == true_label:
            rf_wrong_svc_right.append({
                "sample_index": orig_idx,
                "true_label": int(le.classes_[true_label]),
                "rf_pred": int(le.classes_[rf_p]),
                "svc_pred": int(le.classes_[svc_p])
            })
            
        if svc_p != true_label and rf_p == true_label:
            svc_wrong_rf_right.append({
                "sample_index": orig_idx,
                "true_label": int(le.classes_[true_label]),
                "rf_pred": int(le.classes_[rf_p]),
                "svc_pred": int(le.classes_[svc_p])
            })
            
    analysis = {
        "rf_wrong_svc_right_count": len(rf_wrong_svc_right),
        "rf_wrong_svc_right_samples": rf_wrong_svc_right,
        "svc_wrong_rf_right_count": len(svc_wrong_rf_right),
        "svc_wrong_rf_right_samples": svc_wrong_rf_right
    }
    
    with open(os.path.join(RESULTS_DIR, "ensemble_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=4)
        
    print("Done! Results saved in outputs_for_colab_clean/results_ensemble")

if __name__ == "__main__":
    main()
