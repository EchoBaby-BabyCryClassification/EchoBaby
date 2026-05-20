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

def analyze_classes_uniqueness(root_dir):
    class_map = {} # class_name -> set of hashes
    
    extensions = ('.wav', '.mp3', '.m4a', '.ogg', '.flac')
    
    # Normalize class names (handle spaces vs underscores)
    def normalize_class(name):
        return name.lower().replace(' ', '_')

    for root, dirs, files in os.walk(root_dir):
        dir_name = os.path.basename(root)
        if not dir_name: continue
        
        normalized_class = normalize_class(dir_name)
        
        # Skip top level folders if they don't contain audio directly
        if normalized_class in ['raw_data', 'baby_cry', 'baby_crying_sounds', 'dataset', 'donateacry_corpus']:
            continue
            
        for file in files:
            if file.lower().endswith(extensions):
                file_path = os.path.join(root, file)
                h = get_file_hash(file_path)
                if h:
                    if normalized_class not in class_map:
                        class_map[normalized_class] = set()
                    class_map[normalized_class].add(h)
    
    print(f"{'Class':<20} | {'Unique Files':<12}")
    print("-" * 35)
    
    # Sort by count descending
    sorted_classes = sorted(class_map.items(), key=lambda x: len(x[1]), reverse=True)
    
    for cls, hashes in sorted_classes:
        print(f"{cls:<20} | {len(hashes):<12}")

if __name__ == "__main__":
    analyze_classes_uniqueness("c:\\Users\\ASUS\\Desktop\\aml_project\\raw_data")
