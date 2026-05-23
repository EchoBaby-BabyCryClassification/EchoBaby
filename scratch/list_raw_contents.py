import os

RAW_DATA_1 = r"C:\Users\ASUS\Desktop\aml_project\raw_data"
RAW_DATA_2 = r"C:\Users\ASUS\Desktop\aml_project\raw_data_2"
OUTPUT_FILE = r"C:\Users\ASUS\Desktop\aml_project\raw_data_contents.txt"

def format_size(bytes_size):
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f} KB"
    else:
        return f"{bytes_size / (1024 * 1024):.1f} MB"

def generate_contents_list():
    lines = []
    lines.append("================================================================================")
    lines.append("RAW_DATA VE RAW_DATA_2 KLASÖRLERİ DETAYLI İÇERİK LİSTESİ")
    lines.append("================================================================================")
    lines.append("\nBu dosya projedeki orijinal veri setlerinin (raw_data ve raw_data_2) yapısını ve")
    lines.append("içerdikleri tüm ses dosyalarını hiyerarşik olarak listelemektedir.\n")
    
    # Analyze raw_data
    if os.path.exists(RAW_DATA_1):
        lines.append("================================================================================")
        lines.append("1. KLASÖR: raw_data/")
        lines.append("================================================================================")
        
        file_count_1 = 0
        total_size_1 = 0
        
        for root, dirs, files in os.walk(RAW_DATA_1):
            rel_dir = os.path.relpath(root, RAW_DATA_1)
            level = 0 if rel_dir == "." else len(rel_dir.split(os.sep))
            indent = "  " * level
            
            if rel_dir == ".":
                lines.append("raw_data/ (Kök Dizin)")
            else:
                lines.append(f"{indent}└── {os.path.basename(root)}/")
                
            audio_files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac', '.m4a'))]
            for f in sorted(audio_files):
                file_path = os.path.join(root, f)
                size_str = format_size(os.path.getsize(file_path))
                lines.append(f"{indent}    ├── {f} ({size_str})")
                file_count_1 += 1
                total_size_1 += os.path.getsize(file_path)
                
        lines.append(f"\n--> raw_data ÖZETİ: Toplam {file_count_1} ses dosyası, Toplam Boyut: {format_size(total_size_1)}\n\n")
    else:
        lines.append("raw_data klasörü bulunamadı!\n\n")
        
    # Analyze raw_data_2
    if os.path.exists(RAW_DATA_2):
        lines.append("================================================================================")
        lines.append("2. KLASÖR: raw_data_2/")
        lines.append("================================================================================")
        
        file_count_2 = 0
        total_size_2 = 0
        
        for root, dirs, files in os.walk(RAW_DATA_2):
            rel_dir = os.path.relpath(root, RAW_DATA_2)
            level = 0 if rel_dir == "." else len(rel_dir.split(os.sep))
            indent = "  " * level
            
            if rel_dir == ".":
                lines.append("raw_data_2/ (Kök Dizin)")
            else:
                lines.append(f"{indent}└── {os.path.basename(root)}/")
                
            audio_files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac', '.m4a'))]
            for f in sorted(audio_files):
                file_path = os.path.join(root, f)
                size_str = format_size(os.path.getsize(file_path))
                lines.append(f"{indent}    ├── {f} ({size_str})")
                file_count_2 += 1
                total_size_2 += os.path.getsize(file_path)
                
        lines.append(f"\n--> raw_data_2 ÖZETİ: Toplam {file_count_2} ses dosyası, Toplam Boyut: {format_size(total_size_2)}\n")
    else:
        lines.append("raw_data_2 klasörü bulunamadı!\n")
        
    # Write output to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("\n".join(lines))
        
    print(f"İçerik listesi başarıyla '{OUTPUT_FILE}' dosyasına yazıldı.")

if __name__ == "__main__":
    generate_contents_list()
