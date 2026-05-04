import os
import numpy as np
import pandas as pd

def audit_array_alignment():
    base_dir = r"C:\Users\ASUS\Desktop\aml_project"
    out_dir = os.path.join(base_dir, "outputs_for_colab")
    
    # Load files
    X_img = np.load(os.path.join(out_dir, "X_images.npy"))
    X_tab = np.load(os.path.join(out_dir, "X_tabular.npy"))
    y_lbl = np.load(os.path.join(out_dir, "y_labels.npy"), allow_pickle=True)
    groups = np.load(os.path.join(out_dir, "groups.npy"), allow_pickle=True)
    metadata = pd.read_csv(os.path.join(out_dir, "metadata_final.csv"))
    
    # Consistency check
    mismatches_label = 0
    mismatches_group = 0
    
    for i in range(len(metadata)):
        if metadata.iloc[i].label != y_lbl[i]:
            mismatches_label += 1
        if str(metadata.iloc[i].group_id) != str(groups[i]):
            mismatches_group += 1
            
    with open('audit_array_alignment.md', 'w', encoding='utf-8') as f:
        f.write("# Audit: Array Alignment\n\n")
        f.write(f"Total examples: {len(metadata)}\n")
        f.write(f"Label mismatches (metadata vs y_labels): {mismatches_label}\n")
        f.write(f"Group mismatches (metadata vs groups): {mismatches_group}\n\n")
        
        f.write("## First 30 Samples Alignment Table\n")
        sample_df = metadata.head(30).copy()
        sample_df['y_lbl_value'] = y_lbl[:30]
        sample_df['groups_value'] = groups[:30]
        f.write(sample_df[['index', 'file_path', 'label', 'y_lbl_value', 'group_id', 'groups_value']].to_string() + "\n\n")
        
        f.write("## Feature Integrity\n")
        f.write(f"X_images NaN: {np.isnan(X_img).sum()}\n")
        f.write(f"X_tabular NaN: {np.isnan(X_tab).sum()}\n")
        f.write(f"X_tabular Inf: {np.isinf(X_tab).sum()}\n")

if __name__ == "__main__":
    audit_array_alignment()
