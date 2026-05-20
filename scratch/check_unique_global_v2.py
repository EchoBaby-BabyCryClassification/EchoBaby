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
    
    # Updated exclusion list to include 'not_cry'
    exclude_dirs = {'noise', 'silence', 'laugh', 'not_cry'}
    extensions = ('.wav', '.mp3', '.m4a', '.ogg', '.flac')
    
    filtered_all_files = []
    filtered_hashes = {}
    
    for root, dirs, files in os.walk(root_dir):
        path_parts = set(root.lower().replace('\\', '/').split('/'))
        is_excluded = not path_parts.isdisjoint(exclude_dirs)
        
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
    
    print(f"--- Global Uniqueness Analysis (Including 'Dataset' folder) ---")
    print(f"Total files in raw_data: {len(all_files)}")
    print(f"Unique files in raw_data: {len(hashes)}")
    
    print(f"\n--- Filtered Analysis (Excluding: {', '.join(exclude_dirs)}) ---")
    print(f"Total filtered files: {len(filtered_all_files)}")
    print(f"Unique filtered files: {len(filtered_hashes)}")

if __name__ == "__main__":
    analyze_all_raw_data("c:\\Users\\ASUS\\Desktop\\aml_project\\raw_data")
