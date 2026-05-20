import os
import hashlib
from collections import Counter

RAW_DATA_1 = r"C:\Users\ASUS\Desktop\aml_project\raw_data"
RAW_DATA_2 = r"C:\Users\ASUS\Desktop\aml_project\raw_data_2"

EXTENSIONS = ('.wav', '.mp3', '.m4a', '.ogg', '.flac')

def get_hash(file_path):
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception as e:
        return None

def analyze_raw_data_2():
    lines = []
    lines.append("--- RAW_DATA_2 ANALİZİ ---")
    
    # 1. Klasör bazlı gruplama ve dosya sayımı
    folder_counts = {}
    all_files_2 = []
    hashes_2 = {}
    
    for root, dirs, files in os.walk(RAW_DATA_2):
        rel_path = os.path.relpath(root, RAW_DATA_2)
        if rel_path == '.':
            continue
        top_folder = rel_path.split(os.sep)[0]
        
        for file in files:
            if file.lower().endswith(EXTENSIONS):
                file_path = os.path.join(root, file)
                all_files_2.append(file_path)
                
                h = get_hash(file_path)
                if h:
                    if h not in hashes_2:
                        hashes_2[h] = []
                    hashes_2[h].append(file_path)
                    
                    folder_counts[top_folder] = folder_counts.get(top_folder, 0) + 1
                    
    lines.append("\n[1] raw_data_2 Klasöründeki Dosyaların Gruplanma Durumu:")
    for folder, count in sorted(folder_counts.items()):
        lines.append(f" - {folder}: {count} adet ses dosyası")
        
    # 2. Benzersizlik (Uniqueness) Durumu
    total_files_2 = len(all_files_2)
    unique_files_2 = len(hashes_2)
    duplicate_files_2 = total_files_2 - unique_files_2
    
    lines.append(f"\n[2] raw_data_2 Benzersizlik Durumu:")
    lines.append(f" - Toplam Dosya Sayısı: {total_files_2}")
    lines.append(f" - Benzersiz (Unique) Dosya Sayısı: {unique_files_2}")
    lines.append(f" - Tekrar Eden (Duplicate) Dosya Sayısı: {duplicate_files_2}")
    if total_files_2 > 0:
        lines.append(f" - Tekrar Oranı: {duplicate_files_2 / total_files_2 * 100:.2f}%")
        
    # 3. raw_data (RAW_DATA_1) ile kesişen dosyalar
    lines.append("\n[3] raw_data (Dataset 1) ile Kesişim Analizi yapılıyor...")
    hashes_1 = {}
    all_files_1 = []
    
    for root, dirs, files in os.walk(RAW_DATA_1):
        for file in files:
            if file.lower().endswith(EXTENSIONS):
                file_path = os.path.join(root, file)
                all_files_1.append(file_path)
                h = get_hash(file_path)
                if h:
                    if h not in hashes_1:
                        hashes_1[h] = []
                    hashes_1[h].append(file_path)
                    
    intersection_hashes = set(hashes_2.keys()) & set(hashes_1.keys())
    lines.append(f" - raw_data (Dataset 1) Toplam Dosya Sayısı: {len(all_files_1)}")
    lines.append(f" - raw_data (Dataset 1) Benzersiz Dosya Sayısı: {len(hashes_1)}")
    lines.append(f" - Kesişen (Ortak) Benzersiz Dosya Sayısı (MD5 Hash bazlı): {len(intersection_hashes)}")
    
    # Kesişen dosyaların detayları
    if len(intersection_hashes) > 0:
        lines.append("\nKesişen Dosya Örnekleri (İlk 20 ortak dosya ve konumları):")
        for i, h in enumerate(sorted(list(intersection_hashes))[:20]):
            p1 = [os.path.relpath(p, RAW_DATA_1) for p in hashes_1[h]]
            p2 = [os.path.relpath(p, RAW_DATA_2) for p in hashes_2[h]]
            lines.append(f" Ortak Hash {i+1}: {h}")
            lines.append(f"  -> raw_data içindeki konum(lar): {p1}")
            lines.append(f"  -> raw_data_2 içindeki konum(lar): {p2}")
            lines.append("-" * 50)
    else:
        lines.append("\nraw_data ve raw_data_2 arasında hiç kesişen dosya bulunamadı.")
        
    # Write to UTF-8 file
    out_path = r"C:\Users\ASUS\Desktop\aml_project\scratch\raw_data_2_analysis_results_utf8.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Analysis saved to raw_data_2_analysis_results_utf8.txt")

if __name__ == "__main__":
    analyze_raw_data_2()
