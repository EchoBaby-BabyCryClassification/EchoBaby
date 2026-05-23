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

def find_new_unique_files(root_dir):
    original_dirs = ['BABY CRY', 'Baby Crying Sounds']
    dataset_dir = 'Dataset'
    
    original_hashes = set()
    dataset_hashes_info = {} # hash -> list of paths
    
    extensions = ('.wav', '.mp3', '.m4a', '.ogg', '.flac')
    
    for root, dirs, files in os.walk(root_dir):
        rel_path = os.path.relpath(root, root_dir)
        top_dir = rel_path.split(os.sep)[0]
        
        for file in files:
            if file.lower().endswith(extensions):
                file_path = os.path.join(root, file)
                h = get_file_hash(file_path)
                if not h: continue
                
                if top_dir in original_dirs:
                    original_hashes.add(h)
                elif top_dir == dataset_dir:
                    if h not in dataset_hashes_info:
                        dataset_hashes_info[h] = []
                    dataset_hashes_info[h].append(file_path)
    
    new_hashes = set(dataset_hashes_info.keys()) - original_hashes
    
    print(f"Number of unique hashes found ONLY in 'Dataset' folder: {len(new_hashes)}")
    print("\nList of these files:")
    for i, h in enumerate(list(new_hashes)[:40]):
        paths = dataset_hashes_info[h]
        print(f"{i+1}. {paths[0]}")

if __name__ == "__main__":
    find_new_unique_files("c:\\Users\\ASUS\\Desktop\\aml_project\\raw_data")
