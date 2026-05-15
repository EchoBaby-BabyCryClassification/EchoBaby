import os
import librosa
import soundfile as sf
from pathlib import Path

def process_audio_pipeline():
    # Dizinleri tanımla
    base_dir = r"C:\Users\ASUS\Desktop\aml_project"
    input_dirs = [
        os.path.join(base_dir, "raw_data", "BABY CRY"),
        os.path.join(base_dir, "raw_data", "Baby Crying Sounds")
    ]
    output_dir = os.path.join(base_dir, "standardized_audio")
    
    # Çıktı klasörünü oluştur (yoksa)
    os.makedirs(output_dir, exist_ok=True)
    
    # Desteklenen dosya uzantıları
    valid_extensions = {'.wav', '.ogg', '.mp3'}
    
    # Tüm dosyaları tara ve topla
    audio_files = []
    for input_dir in input_dirs:
        if os.path.exists(input_dir):
            for root, _, files in os.walk(input_dir):
                for file in files:
                    ext = Path(file).suffix.lower()
                    if ext in valid_extensions:
                        audio_files.append(os.path.join(root, file))
        else:
            print(f"Uyarı: Dizin bulunamadı - {input_dir}")
            
    total_files = len(audio_files)
    print(f"Toplam {total_files} adet ses dosyası bulundu.\n")
    
    success_count = 0
    
    for idx, file_path in enumerate(audio_files, 1):
        try:
            # Orijinal dosya adını uzantısız olarak al
            filename = Path(file_path).stem
            output_path = os.path.join(output_dir, f"{filename}.wav")
            
            print(f"[{idx}/{total_files}] İşleniyor: {filename}{Path(file_path).suffix} ...")
            
            # 1. librosa.load: Dosyayı yüklerken otomatik olarak 16000 Hz'e, Mono'ya ve float32'ye dönüştürür
            y, sr = librosa.load(file_path, sr=16000, mono=True)
            
            # 2. Seslerin başındaki ve sonundaki sessizliği temizle
            y_trimmed, _ = librosa.effects.trim(y, top_db=30)
            
            # 3. float32 formatında .wav olarak kaydet
            sf.write(output_path, y_trimmed, sr, subtype='FLOAT')
            
            success_count += 1
            
        except Exception as e:
            print(f"[{idx}/{total_files}] Hata oluştu ({file_path}): {e}")
            
    # Sonuç raporu
    print("\n" + "="*40)
    print("İŞLEM TAMAMLANDI")
    print(f"Başarıyla dönüştürülen dosya sayısı: {success_count} / {total_files}")
    print("="*40)

if __name__ == "__main__":
    process_audio_pipeline()
