import os
import hashlib

def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def analyze_filtered_audio(root_dir):
    all_files = []
    hashes = {}
    
    exclude_dirs = {'noise', 'silence', 'laugh'}
    extensions = ('.wav', '.mp3', '.m4a', '.ogg', '.flac')
    
    for root, dirs, files in os.walk(root_dir):
        # Check if current directory should be excluded
        dir_name = os.path.basename(root).lower()
        if dir_name in exclude_dirs:
            continue
            
        for file in files:
            if file.lower().endswith(extensions):
                file_path = os.path.join(root, file)
                
                # Check if any part of the path contains the excluded categories (just to be safe)
                path_parts = set(root.lower().split(os.sep))
                if any(ex in path_parts for ex in exclude_dirs):
                    continue
                    
                all_files.append(file_path)
                
                file_hash = get_file_hash(file_path)
                if file_hash not in hashes:
                    hashes[file_hash] = []
                hashes[file_hash].append(file_path)
    
    total_files = len(all_files)
    unique_files = len(hashes)
    
    print(f"--- Filtered Results (Excluding noise, silence, laugh) ---")
    print(f"Total audio files: {total_files}")
    print(f"Unique audio files: {unique_files}")
    
    with open("c:\\Users\\ASUS\\Desktop\\aml_project\\scratch\\filtered_analysis_summary.txt", "w") as f:
        f.write(f"Total audio files (filtered): {total_files}\n")
        f.write(f"Unique audio files (filtered): {unique_files}\n")

if __name__ == "__main__":
    analyze_filtered_audio("c:\\Users\\ASUS\\Desktop\\aml_project\\raw_data")
