import os
import numpy as np
import pandas as pd

def audit_group_quality():
    out_dir = r"C:\Users\ASUS\Desktop\aml_project\outputs_for_colab"
    y_lbl = np.load(os.path.join(out_dir, "y_labels.npy"), allow_pickle=True)
    groups = np.load(os.path.join(out_dir, "groups.npy"), allow_pickle=True)
    
    df = pd.DataFrame({'label': y_lbl, 'group_id': groups})
    
    group_stats = df.groupby('group_id').agg({
        'label': ['count', 'nunique', 'unique']
    }).reset_index()
    group_stats.columns = ['group_id', 'sample_count', 'unique_labels', 'labels']
    
    mixed_groups = group_stats[group_stats['unique_labels'] > 1]
    
    with open('audit_group_quality.md', 'w', encoding='utf-8') as f:
        f.write("# Audit: Group Quality\n\n")
        f.write(f"Total Groups: {len(group_stats)}\n")
        f.write(f"Mixed Groups (Groups with >1 label): {len(mixed_groups)}\n\n")
        
        if len(mixed_groups) > 0:
            f.write("## Mixed Groups Details\n")
            f.write(mixed_groups.to_string() + "\n\n")
        
        f.write("## Sample Count Distribution per Group\n")
        f.write(group_stats['sample_count'].value_counts().sort_index().to_string() + "\n")

if __name__ == "__main__":
    audit_group_quality()
