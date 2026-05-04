import os
import pandas as pd
from pathlib import Path

def audit_csv_alignment():
    base_dir = r"C:\Users\ASUS\Desktop\aml_project"
    csv_path = os.path.join(base_dir, "raw_data", "donateacry-corpus_features_final.csv")
    
    if not os.path.exists(csv_path):
        print("CSV not found")
        return
        
    df = pd.read_csv(csv_path)
    
    # Inventory listesinden hash bilgilerini al
    inv_path = 'audit_raw_inventory.csv'
    if not os.path.exists(inv_path):
        print("Inventory CSV not found, run audit_raw_inventory.py first")
        return
    inv_df = pd.read_csv(inv_path)
    
    # Matching logic
    df['filename'] = df['Cry_Audio_File'].apply(lambda x: Path(x).name)
    
    matches = pd.merge(df, inv_df, on='filename', how='left', indicator=True)
    
    matched = matches[matches['_merge'] == 'both']
    unmatched_csv = matches[matches['_merge'] == 'left_only']
    
    with open('audit_csv_alignment.md', 'w', encoding='utf-8') as f:
        f.write("# Audit: CSV Alignment\n\n")
        f.write(f"CSV Rows: {len(df)}\n")
        f.write(f"Matched with raw files: {len(matched)}\n")
        f.write(f"Unmatched CSV entries: {len(unmatched_csv)}\n\n")
        
        if len(unmatched_csv) > 0:
            f.write("## Unmatched Examples (First 10)\n")
            f.write(unmatched_csv[['filename', 'Cry_Audio_File']].head(10).to_string() + "\n\n")
            
        f.write("## Label Consistency (CSV vs Folder)\n")
        if len(matched) > 0:
            consistency = matched.groupby(['Cry_Reason', 'parent']).size().reset_index(name='count')
            f.write(consistency.to_string() + "\n\n")

if __name__ == "__main__":
    audit_csv_alignment()
