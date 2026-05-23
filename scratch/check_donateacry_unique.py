import os
import hashlib

def get_file_hash(file_path):
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except:
        return None

def analyze_donateacry(root_dir):
    all_files = []
    hashes = {}
    extensions = ('.wav', '.mp3', '.m4a', '.ogg', '.flac')
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(extensions):
                file_path = os.path.join(root, file)
                h = get_file_hash(file_path)
                if h:
                    all_files.append(file_path)
                    if h not in hashes:
                        hashes[h] = []
                    hashes[h].append(file_path)
    
    print(f"--- Analysis for 'donateacry_corpus' ---")
    print(f"Total audio files: {len(all_files)}")
    print(f"Unique audio files: {len(hashes)}")
    print(f"Duplicates within this folder: {len(all_files) - len(hashes)}")

if __name__ == "__main__":
    analyze_donateacry("c:\\Users\\ASUS\\Desktop\\aml_project\\raw_data\\donateacry_corpus")
