import os
import hashlib

def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def analyze_audio_files(root_dir):
    all_files = []
    hashes = {}
    extensions = ('.wav', '.mp3', '.m4a', '.ogg', '.flac')
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(extensions):
                file_path = os.path.join(root, file)
                all_files.append(file_path)
                file_hash = get_file_hash(file_path)
                if file_hash not in hashes:
                    hashes[file_hash] = []
                hashes[file_hash].append(file_path)
    
    total_files = len(all_files)
    unique_files = len(hashes)
    duplicates = total_files - unique_files
    
    with open("c:\\Users\\ASUS\\Desktop\\aml_project\\scratch\\analysis_summary.txt", "w") as f:
        f.write(f"Total audio files: {total_files}\n")
        f.write(f"Unique audio files (by content): {unique_files}\n")
        f.write(f"Duplicate files: {duplicates}\n")

if __name__ == "__main__":
    analyze_audio_files("c:\\Users\\ASUS\\Desktop\\aml_project\\raw_data")
