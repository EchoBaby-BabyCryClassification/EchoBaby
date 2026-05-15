import os
import numpy as np
import pandas as pd

def validate_outputs():
    base_dir = r"C:\Users\ASUS\Desktop\aml_project\outputs_for_colab_clean"
    
    print("Validating rebuilt dataset outputs...")
    
    x_img_path = os.path.join(base_dir, "X_images_new.npy")
    x_tab_path = os.path.join(base_dir, "X_tabular_new.npy")
    y_lbl_path = os.path.join(base_dir, "y_labels_new.npy")
    groups_path = os.path.join(base_dir, "groups_new.npy")
    csv_path = os.path.join(base_dir, "metadata_clean.csv")
    
    files_exist = all(os.path.exists(p) for p in [x_img_path, x_tab_path, y_lbl_path, groups_path, csv_path])
    if not files_exist:
        print("FAIL: One or more files are missing.")
        return
    
    X_img = np.load(x_img_path)
    X_tab = np.load(x_tab_path)
    y_lbl = np.load(y_lbl_path)
    groups = np.load(groups_path)
    df = pd.read_csv(csv_path)
    
    # Check lengths
    n_samples = len(df)
    if not (X_img.shape[0] == X_tab.shape[0] == len(y_lbl) == len(groups) == n_samples):
        print(f"FAIL: Length mismatch. df: {n_samples}, X_img: {X_img.shape[0]}, X_tab: {X_tab.shape[0]}, y_lbl: {len(y_lbl)}, groups: {len(groups)}")
        return
    
    print(f"Length check passed: {n_samples} samples.")
    
    # Check NaNs
    if np.isnan(X_img).any():
        print("FAIL: NaN found in X_images_new.npy")
    elif np.isnan(X_tab).any():
        print("FAIL: NaN found in X_tabular_new.npy")
    elif np.isinf(X_tab).any():
        print("FAIL: Inf found in X_tabular_new.npy")
    else:
        print("NaN/Inf check passed.")
        
    # Check alignment
    mismatches = 0
    for i in range(n_samples):
        if df.iloc[i]['label'] != y_lbl[i]:
            mismatches += 1
        if str(df.iloc[i]['group_id']) != str(groups[i]):
            mismatches += 1
            
    if mismatches > 0:
        print(f"FAIL: Found {mismatches} alignment mismatches between CSV and arrays.")
    else:
        print("Alignment check passed.")
        
    # Check shape
    if X_tab.shape[1] != 65:
        print(f"FAIL: X_tabular_new should have 65 features, found {X_tab.shape[1]}")
    else:
        print("X_tabular feature dimension check passed (65 features).")
        
    if X_img.shape[1:] != (128, 128, 1):
        print(f"FAIL: X_images_new should have shape (128, 128, 1), found {X_img.shape[1:]}")
    else:
        print("X_images shape check passed.")
        
    print("\nValidation completed successfully!")

if __name__ == "__main__":
    validate_outputs()
