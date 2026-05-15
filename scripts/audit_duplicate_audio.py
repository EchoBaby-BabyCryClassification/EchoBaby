import os
import pandas as pd

def audit_duplicate_audio():
    inv_path = 'audit_raw_inventory.csv'
    if not os.path.exists(inv_path):
        print("Inventory CSV not found")
        return
        
    df = pd.read_csv(inv_path)
    
    # Hash duplication
    duplicates = df[df.duplicated(subset=['hash'], keep=False)].sort_values('hash')
    
    with open('audit_duplicate_audio.md', 'w', encoding='utf-8') as f:
        f.write("# Audit: Duplicate Audio Analysis\n\n")
        f.write(f"Total files: {len(df)}\n")
        f.write(f"Files with identical hashes: {len(duplicates)}\n\n")
        
        if len(duplicates) > 0:
            f.write("## Exact Duplicates (by MD5 hash)\n")
            f.write(duplicates[['hash', 'rel_path', 'parent', 'duration_sec']].to_string() + "\n\n")
            
            # Cross-label duplicates
            hash_parents = duplicates.groupby('hash')['parent'].nunique()
            cross_label = hash_parents[hash_parents > 1].index
            
            if len(cross_label) > 0:
                f.write("## CROSS-LABEL DUPLICATES (CRITICAL)\n")
                f.write(duplicates[duplicates['hash'].isin(cross_label)].to_string() + "\n\n")
        
        # Near duplicates by filename stem
        df['stem'] = df['filename'].apply(lambda x: os.path.splitext(x)[0])
        stem_dupes = df[df.duplicated(subset=['stem'], keep=False)].sort_values('stem')
        
        f.write("## Filename Stem Duplicates\n")
        f.write(f"Total stem duplicates: {len(stem_dupes)}\n")
        if len(stem_dupes) > 0:
            f.write(stem_dupes[['stem', 'rel_path', 'ext', 'hash']].head(50).to_string() + "\n")

if __name__ == "__main__":
    audit_duplicate_audio()
