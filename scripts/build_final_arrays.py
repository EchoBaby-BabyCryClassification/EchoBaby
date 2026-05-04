import os
import pandas as pd
import numpy as np
import re
import shutil

def extract_group_id(filename):
    match = re.match(r'^([a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12})', filename)
    if match:
        return match.group(1).upper()
    else:
        return filename.split('-')[0].upper()

def build_final_arrays():
    base_dir = r"C:\Users\ASUS\Desktop\aml_project"
    out_dir = os.path.join(base_dir, "outputs_for_colab")
    
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(base_dir, "final_feature_matrix_v2.csv")
    x_img_path = os.path.join(base_dir, "X_images.npy")
    x_tab_path = os.path.join(base_dir, "X_tabular.npy")
    y_lbl_path = os.path.join(base_dir, "y_labels.npy")
    
    print("Loading data...")
    df = pd.read_csv(csv_path)
    X_img = np.load(x_img_path)
    X_tab = np.load(x_tab_path)
    y_lbl = np.load(y_lbl_path)
    
    if not (len(df) == len(X_img) == len(X_tab) == len(y_lbl)):
        print("ERROR: Lengths of files do not match!")
        return
        
    print(f"Data length: {len(df)}")
    
    # Generate groups
    df['group_id'] = df['filename'].apply(extract_group_id)
    groups = df['group_id'].values
    
    # Save groups.npy
    out_groups_path = os.path.join(out_dir, "groups.npy")
    np.save(out_groups_path, groups)
    
    # Create metadata_final.csv
    metadata_df = pd.DataFrame({
        'index': range(len(df)),
        'file_path': df['filename'],
        'label': df['Cry_Reason'],
        'group_id': df['group_id'],
        'source_filename': df['filename'],
        'split_candidate_note': 'Derived from UUID pattern'
    })
    
    out_meta_path = os.path.join(out_dir, "metadata_final.csv")
    metadata_df.to_csv(out_meta_path, index=False)
    
    # Copy NPY files to output folder
    out_x_img_path = os.path.join(out_dir, "X_images.npy")
    out_x_tab_path = os.path.join(out_dir, "X_tabular.npy")
    out_y_lbl_path = os.path.join(out_dir, "y_labels.npy")
    
    np.save(out_x_img_path, X_img)
    np.save(out_x_tab_path, X_tab)
    np.save(out_y_lbl_path, y_lbl)
    
    # Generate Report
    report_path = os.path.join(out_dir, "dataset_report.md")
    
    group_counts = df['group_id'].value_counts()
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# AML Project - Dataset Final Report\n\n")
        f.write(f"- **Toplam örnek sayısı**: {len(df)}\n")
        f.write(f"- **X_images shape**: {X_img.shape}\n")
        f.write(f"- **X_tabular shape**: {X_tab.shape}\n")
        f.write(f"- **y_labels shape**: {y_lbl.shape}\n")
        f.write(f"- **groups shape**: {groups.shape}\n\n")
        
        f.write("## Sınıf Dağılımı\n")
        f.write(df['Cry_Reason'].value_counts().to_string() + "\n\n")
        
        f.write("## Grup (Subject) İstatistikleri\n")
        f.write(f"- **Grup sayısı**: {len(group_counts)}\n")
        f.write(f"- **Grup başına ortalama örnek sayısı**: {group_counts.mean():.2f}\n\n")
        
        f.write("### En büyük 20 grup\n")
        f.write(group_counts.head(20).to_string() + "\n\n")
        
        f.write("## Leakage ve Split Analizi\n")
        f.write("Aynı `group_id`'ye (UUID) sahip örnekler, orijinal ses dosyasından bölünen veya aynı kayıt cihazından aynı seansta kaydedilen farklı ses dosyalarıdır. "
                "Bu grupları böldüğümüzde Data Leakage tamamen engellenecektir. `groups.npy` bu yüzden oluşturuldu.\n\n")
        f.write("**Kullanılan group_id çıkarım mantığı**: `[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-...` formatındaki UUID yapısı her bir dosya isminin başından regex ile çıkarılarak Subject/Session kimliği elde edilmiştir.\n")

    print("All final arrays and report built successfully in outputs_for_colab/")

if __name__ == "__main__":
    build_final_arrays()
