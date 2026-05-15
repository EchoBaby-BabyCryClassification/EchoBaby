import os
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
import warnings
warnings.filterwarnings('ignore')

def main():
    BASE_DIR = r"C:\Users\ASUS\Desktop\aml_project\outputs_for_colab_clean"
    RESULTS_DIR = os.path.join(BASE_DIR, "results_resampling")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Loading V2 data...")
    X_tab = np.load(os.path.join(BASE_DIR, "X_tabular_v2.npy"))
    y_raw = np.load(os.path.join(BASE_DIR, "y_labels_new.npy"), allow_pickle=True)
    groups = np.load(os.path.join(BASE_DIR, "groups_new.npy"), allow_pickle=True)
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    target_names = [str(c) for c in le.classes_]
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    def get_model():
        # We don't use class_weight="balanced" here because we are resampling to balance it explicitly
        return ExtraTreesClassifier(random_state=42)
        
    scenarios = {
        "A_No_Resampling": None,
        "B_RandomOverSampler": RandomOverSampler(random_state=42),
        "C_SMOTE": SMOTE(random_state=42, k_neighbors=3) # set low to avoid error if a fold splits bad
    }
    
    results = {s: {"macro_f1": [], "per_class_recall": {c: [] for c in target_names}} for s in scenarios}
    
    for scenario_name, sampler in scenarios.items():
        print(f"\nEvaluating Scenario: {scenario_name}...")
        
        for train_idx, test_idx in sgkf.split(X_tab, y, groups):
            X_train, X_test = X_tab[train_idx], X_tab[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Resample only train data!
            if sampler is not None:
                try:
                    X_train, y_train = sampler.fit_resample(X_train, y_train)
                except Exception as e:
                    print(f"Sampling failed for {scenario_name}: {e}")
                    continue
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            model = get_model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            results[scenario_name]["macro_f1"].append(f1)
            
            recalls = recall_score(y_test, y_pred, average=None, labels=np.arange(len(target_names)), zero_division=0)
            for i, c in enumerate(target_names):
                results[scenario_name]["per_class_recall"][c].append(recalls[i])
                
    summary = {}
    for s in scenarios:
        if len(results[s]["macro_f1"]) > 0:
            summary[s] = {
                "macro_f1_mean": float(np.mean(results[s]["macro_f1"])),
            }
            for c in target_names:
                summary[s][f"recall_class_{c}_mean"] = float(np.mean(results[s]["per_class_recall"][c]))
                
    with open(os.path.join(RESULTS_DIR, "resampling_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print("\n" + "="*40)
    print("RESULTS SUMMARY")
    print("="*40)
    for s, metrics in summary.items():
        print(f"\n--- {s} ---")
        print(f"Macro F1: {metrics['macro_f1_mean']:.4f}")
        for c in target_names:
            print(f"  Class {c} Recall: {metrics[f'recall_class_{c}_mean']:.4f}")

if __name__ == "__main__":
    main()
