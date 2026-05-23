import os

def check_extensions(root_dir):
    extensions_count = {}
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext:
                extensions_count[ext] = extensions_count.get(ext, 0) + 1
    
    print("Extension counts in all of raw_data:")
    for ext, count in extensions_count.items():
        print(f"{ext}: {count}")

if __name__ == "__main__":
    check_extensions("c:\\Users\\ASUS\\Desktop\\aml_project\\raw_data")
