import os
import hashlib

def get_file_hash(file_path):
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception as e:
        return None

def analyze_all_raw_data(root_dir):
    all_files = []
    hashes = {}
    
    exclude_dirs = {'noise', 'silence', 'laugh'}
    extensions = ('.wav', '.mp3', '.m4a', '.ogg', '.flac')
    
    # For filtered counts
    filtered_all_files = []
    filtered_hashes = {}
    
    for root, dirs, files in os.walk(root_dir):
        dir_name = os.path.basename(root).lower()
        is_excluded = dir_name in exclude_dirs or any(ex in root.lower().split(os.sep) for ex in exclude_dirs)
        
        for file in files:
            if file.lower().endswith(extensions):
                file_path = os.path.join(root, file)
                file_hash = get_file_hash(file_path)
                
                if file_hash:
                    all_files.append(file_path)
                    if file_hash not in hashes:
                        hashes[file_hash] = []
                    hashes[file_hash].append(file_path)
                    
                    if not is_excluded:
                        filtered_all_files.append(file_path)
                        if file_hash not in filtered_hashes:
                            filtered_hashes[file_hash] = []
                        filtered_hashes[file_hash].append(file_path)
    
    total_files = len(all_files)
    unique_files = len(hashes)
    
    total_filtered = len(filtered_all_files)
    unique_filtered = len(filtered_hashes)
    
    print(f"--- Global Uniqueness Analysis (Including 'Dataset' folder) ---")
    print(f"Total files in raw_data: {total_files}")
    print(f"Unique files in raw_data: {unique_files}")
    print(f"Duplicates: {total_files - unique_files}")
    
    print(f"\n--- Filtered Analysis (Excluding noise, silence, laugh) ---")
    print(f"Total filtered files: {total_filtered}")
    print(f"Unique filtered files: {unique_filtered}")
    print(f"Filtered Duplicates: {total_filtered - unique_filtered}")

if __name__ == "__main__":
    analyze_all_raw_data("c:\\Users\\ASUS\\Desktop\\aml_project\\raw_data")
