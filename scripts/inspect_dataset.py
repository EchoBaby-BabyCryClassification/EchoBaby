import os
import pandas as pd
import numpy as np
import re
from collections import Counter

def extract_group_id(filename):
    # UUID pattern matching (e.g. 999bf14b-e417-4b44-b746-9253f81efe38)
    match = re.match(r'^([a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12})', filename)
    if match:
        return match.group(1).upper()
    else:
        # Fallback if no UUID found
        return filename.split('-')[0].upper()

def inspect_dataset():
    base_dir = r"C:\Users\ASUS\Desktop\aml_project"
    csv_path = os.path.join(base_dir, "final_feature_matrix_v2.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} bulunamadı.")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Total entries in CSV: {len(df)}")
    
    df['group_id'] = df['filename'].apply(extract_group_id)
    
    print("\n--- Sınıf (Label) Dağılımı ---")
    print(df['Cry_Reason'].value_csv() if hasattr(df['Cry_Reason'], 'value_csv') else df['Cry_Reason'].value_counts())
    
    print("\n--- Grup (Subject) Dağılımı ---")
    group_counts = df['group_id'].value_counts()
    print(f"Toplam farklı grup sayısı: {len(group_counts)}")
    print(f"Grup başına ortalama örnek sayısı: {group_counts.mean():.2f}")
    print(f"En çok örnek içeren 5 grup:\n{group_counts.head(5)}")
    
    print("\n--- Leakage (Sızıntı) Analizi ---")
    # Eğer bir grup birden fazla sınıfa (Cry_Reason) sahipse, bu bir anomalidir (bebek aynı anda farklı sebeplerden ağlayabilir ama dataset genelde saf tutulur)
    group_class_counts = df.groupby('group_id')['Cry_Reason'].nunique()
    multi_class_groups = group_class_counts[group_class_counts > 1]
    if len(multi_class_groups) > 0:
        print(f"UYARI: {len(multi_class_groups)} adet grup birden fazla sınıfa atanmış. Bu normal olabilir fakat incelenmeli.")
    else:
        print("Tüm gruplar tek bir sınıfa ait (Temiz).")
        
    print("\nDenetim başarıyla tamamlandı.")

if __name__ == "__main__":
    inspect_dataset()
