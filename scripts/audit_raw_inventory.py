import os
import hashlib
import pandas as pd
from pathlib import Path
import librosa
import numpy as np

def get_hash(f):
    hasher = hashlib.md5()
    with open(f, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()

def check_raw_inventory():
    base_dir = r"C:\Users\ASUS\Desktop\aml_project\raw_data"
    data = []
    
    # Scan BABY CRY and Baby Crying Sounds
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.ogg', '.mp3')):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_dir)
                parent = Path(file_path).parent.name
                
                try:
                    # Sadece meta datayı oku
                    y, sr = librosa.load(file_path, sr=None, duration=0.1) # Hızlı olsun diye
                    duration = librosa.get_duration(path=file_path)
                    channels = 1 if len(y.shape) == 1 else y.shape[1]
                except Exception as e:
                    print(f"Error reading {file}: {e}")
                    y, sr, duration, channels = None, None, 0, 0
                
                data.append({
                    'rel_path': rel_path,
                    'filename': file,
                    'ext': Path(file).suffix,
                    'parent': parent,
                    'duration_sec': duration,
                    'sr': sr,
                    'channels': channels,
                    'hash': get_hash(file_path)
                })
                
    df = pd.DataFrame(data)
    df.to_csv('audit_raw_inventory.csv', index=False)
    
    # Report summary
    with open('audit_raw_inventory.md', 'w', encoding='utf-8') as f:
        f.write("# Audit: Raw Data Inventory\n\n")
        f.write(f"Total files found: {len(df)}\n\n")
        f.write("## By Parent Folder\n")
        f.write(df['parent'].value_counts().to_string() + "\n\n")
        f.write("## By Extension\n")
        f.write(df['ext'].value_counts().to_string() + "\n\n")
        
        duplicates = df[df.duplicated(subset=['hash'], keep=False)]
        f.write(f"## Hash Duplicates\nTotal duplicate files (by hash): {len(duplicates)}\n")
        if len(duplicates) > 0:
            f.write(duplicates.sort_values('hash').head(20).to_string() + "\n\n")

if __name__ == "__main__":
    check_raw_inventory()
