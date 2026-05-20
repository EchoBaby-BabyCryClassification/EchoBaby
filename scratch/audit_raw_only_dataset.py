import os
import hashlib
import pandas as pd
import numpy as np
from collections import Counter
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Try importing soundfile and librosa
try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None

WORKSPACE_DIR = r"C:\Users\ASUS\Desktop\aml_project"
AUDIT_DIR = os.path.join(WORKSPACE_DIR, "raw_only_dataset_audit")
os.makedirs(AUDIT_DIR, exist_ok=True)

SUPPORTED_EXTENSIONS = ('.wav', '.mp3', '.ogg', '.flac', '.m4a')

# Mapping rules
FIVE_CLASS_MAP = {
    'belly pain': 'Belly Pain', 'belly_pain': 'Belly Pain', 'b_pain': 'Belly Pain', 'stomach pain': 'Belly Pain', 'colic': 'Belly Pain', 'eairh': 'Belly Pain',
    'burping': 'Burping', 'burp': 'Burping', 'need burping': 'Burping', 'eh': 'Burping',
    'discomfort': 'Discomfort', 'uncomfortable': 'Discomfort', 'cold': 'Discomfort', 'hot': 'Discomfort', 'cold_hot': 'Discomfort', 'cold-hot': 'Discomfort', 'heh': 'Discomfort',
    'hungry': 'Hungry', 'hunger': 'Hungry', 'neh': 'Hungry',
    'tired': 'Tired', 'tiredness': 'Tired', 'sleepy': 'Tired', 'owh': 'Tired'
}

EXCLUDE_LABELS = ['silence', 'noise', 'laugh', 'non-cry', 'non crying', 'snoring', 'not_cry', 'other']

def get_md5_hash(file_path):
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception as e:
        return None

def infer_raw_label_from_filename(filename):
    fn = filename.lower()
    if "belly_pain" in fn or "belly pain" in fn or "colic" in fn or "stomach" in fn or "b_pain" in fn or "eairh" in fn:
        return "belly_pain"
    elif "burping" in fn or "burp" in fn or "eh" in fn:
        return "burping"
    elif "discomfort" in fn or "uncomfortable" in fn or "cold_hot" in fn or "cold-hot" in fn or "heh" in fn or "cold" in fn or "hot" in fn:
        return "discomfort"
    elif "hungry" in fn or "hunger" in fn or "neh" in fn:
        return "hungry"
    elif "tired" in fn or "sleepy" in fn or "owh" in fn or "tiredness" in fn:
        return "tired"
    elif any(ex in fn for ex in EXCLUDE_LABELS) or "snore" in fn:
        return "silence"
    else:
        return "unknown"

def get_audio_metadata(file_path):
    readable = "no"
    duration = None
    sr = None
    channels = None
    
    # Try soundfile
    if sf is not None:
        try:
            info = sf.info(file_path)
            readable = "yes"
            duration = info.duration
            sr = info.samplerate
            channels = info.channels
            return readable, duration, sr, channels
        except Exception:
            pass
            
    # Try librosa next
    if librosa is not None:
        try:
            y, rate = librosa.load(file_path, sr=None, duration=0.1)
            readable = "yes"
            sr = rate
            duration = librosa.get_duration(y=y, sr=rate)
            channels = 1
            return readable, duration, sr, channels
        except Exception:
            pass
            
    return readable, duration, sr, channels

def map_to_5_class(raw_label):
    if not raw_label:
        return "Unknown"
    rl = str(raw_label).lower().strip()
    if rl in FIVE_CLASS_MAP:
        return FIVE_CLASS_MAP[rl]
    elif rl in EXCLUDE_LABELS:
        return "Exclude"
    else:
        return "Unknown"

def map_to_binary(five_class_label):
    if five_class_label == "Hungry":
        return "Hungry"
    elif five_class_label in ["Belly Pain", "Burping", "Discomfort", "Tired"]:
        return "Not Hungry"
    elif five_class_label == "Exclude":
        return "Exclude"
    else:
        return "Unknown"

def main():
    print("Starting raw-only dataset audit...")
    inventory = []
    
    # Only scan raw_data/ and raw_data_2/
    target_dirs = ["raw_data", "raw_data_2"]
    
    for td in target_dirs:
        root_path = os.path.join(WORKSPACE_DIR, td)
        if not os.path.exists(root_path):
            print(f"Skipping {td} because it does not exist.")
            continue
            
        print(f"Scanning {td}...")
        for root, dirs, files in os.walk(root_path):
            rel_dir = os.path.relpath(root, root_path)
            parts = rel_dir.split(os.sep)
            source_top_folder = parts[0] if parts[0] != '.' else '.'
            
            for file in files:
                if file.lower().endswith(SUPPORTED_EXTENSIONS):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, WORKSPACE_DIR)
                    
                    # Size and MD5
                    file_size = os.path.getsize(file_path)
                    md5_val = get_md5_hash(file_path)
                    
                    # Labels
                    parent_folder = os.path.basename(root)
                    inferred_folder = parent_folder.lower()
                    inferred_filename = infer_raw_label_from_filename(file)
                    
                    # Audio check
                    readable, duration, sr, channels = get_audio_metadata(file_path)
                    
                    # Map
                    c5_from_folder = map_to_5_class(inferred_folder)
                    c5_from_file = map_to_5_class(inferred_filename)
                    
                    if c5_from_folder != "Unknown":
                        c5_candidate = c5_from_folder
                    elif c5_from_file != "Unknown":
                        c5_candidate = c5_from_file
                    else:
                        c5_candidate = "Unknown"
                        
                    cb_candidate = map_to_binary(c5_candidate)
                    
                    inventory.append({
                        'source_root': td,
                        'source_top_folder': source_top_folder,
                        'relative_path': rel_path,
                        'filename': file,
                        'extension': os.path.splitext(file)[1].lower(),
                        'file_size_bytes': file_size,
                        'md5_hash': md5_val,
                        'inferred_raw_label_from_folder': parent_folder,
                        'inferred_raw_label_from_filename': inferred_filename,
                        'readable_audio': readable,
                        'duration_seconds': duration,
                        'sample_rate': sr,
                        'channels': channels,
                        'candidate_5_class': c5_candidate,
                        'candidate_binary': cb_candidate
                    })
                    
    df_inv = pd.DataFrame(inventory)
    print(f"Total raw-only files audited: {len(df_inv)}")
    
    # Save inventory
    df_inv.to_csv(os.path.join(AUDIT_DIR, "raw_only_audio_inventory.csv"), index=False)
    
    # Analyze by MD5 hash group
    hash_groups = []
    conflict_report = []
    
    # We will build separate strict and majority candidates
    grouped = df_inv.groupby('md5_hash')
    
    for md5_val, group in grouped:
        copies_count = len(group)
        paths = group['relative_path'].tolist()
        inferred_folders = group['inferred_raw_label_from_folder'].tolist()
        inferred_filenames = group['inferred_raw_label_from_filename'].tolist()
        c5_classes = group['candidate_5_class'].tolist()
        cb_classes = group['candidate_binary'].tolist()
        
        unique_paths = "; ".join(paths)
        unique_inferred_labels = list(set(inferred_folders + inferred_filenames))
        unique_c5 = list(set(c5_classes))
        unique_cb = list(set(cb_classes))
        
        # 5-class active labels (exclude Unknown and Exclude)
        active_5c = [c for c in c5_classes if c not in ["Unknown", "Exclude"]]
        set_active_5c = set(active_5c)
        
        # Strict 5-class candidate criteria
        # Qualifies if there is exactly 1 mapped label and NO conflict (i.e. all copies have same label or some are unknown, but only one active class exists)
        is_strict_5class = False
        strict_5class_label = "Excluded/Unknown"
        if len(set_active_5c) == 1:
            strict_5class_label = list(set_active_5c)[0]
            is_strict_5class = True
            
        # Majority 5-class candidate criteria
        # Qualifies if active label count > 0 and there is a clear majority winner
        is_majority_5class = False
        majority_5class_label = "Excluded/Unknown"
        majority_resolved_by_voting = False
        
        if len(active_5c) > 0:
            c = Counter(active_5c)
            most_common = c.most_common(2)
            if len(most_common) == 1:
                # Only one class present
                majority_5class_label = most_common[0][0]
                is_majority_5class = True
                majority_resolved_by_voting = False
            else:
                # Check for tie
                if most_common[0][1] > most_common[1][1]:
                    majority_5class_label = most_common[0][0]
                    is_majority_5class = True
                    majority_resolved_by_voting = True
                else:
                    # Tie
                    majority_5class_label = "Unresolved Conflict"
                    is_majority_5class = False
                    
        # Binary active labels (exclude Unknown and Exclude)
        active_cb = [c for c in cb_classes if c not in ["Unknown", "Exclude"]]
        set_active_cb = set(active_cb)
        
        # Strict binary
        is_strict_binary = False
        strict_binary_label = "Excluded/Unknown"
        if len(set_active_cb) == 1:
            strict_binary_label = list(set_active_cb)[0]
            is_strict_binary = True
            
        # Majority binary
        is_majority_binary = False
        majority_binary_label = "Excluded/Unknown"
        majority_binary_resolved_by_voting = False
        
        if len(active_cb) > 0:
            c = Counter(active_cb)
            most_common = c.most_common(2)
            if len(most_common) == 1:
                majority_binary_label = most_common[0][0]
                is_majority_binary = True
                majority_binary_resolved_by_voting = False
            else:
                if most_common[0][1] > most_common[1][1]:
                    majority_binary_label = most_common[0][0]
                    is_majority_binary = True
                    majority_binary_resolved_by_voting = True
                else:
                    majority_binary_label = "Unresolved Conflict"
                    is_majority_binary = False
                    
        # Conflict status for 5-class
        if len(set_active_5c) > 1:
            conflict_status = "label_conflict"
            conflict_report.append({
                'md5_hash': md5_val,
                'copies_count': copies_count,
                'file_paths': paths,
                'c5_classes': c5_classes,
                'resolved_5_class': majority_5class_label if is_majority_5class else "Unresolved Conflict"
            })
        elif len(set_active_5c) == 1:
            conflict_status = "no_conflict"
        else:
            conflict_status = "exclude_candidate" if "Exclude" in c5_classes else "unknown_label"
            
        hash_groups.append({
            'md5_hash': md5_val,
            'copies_count': copies_count,
            'file_paths': unique_paths,
            'inferred_raw_labels': "; ".join(unique_inferred_labels),
            'candidate_5_class_labels': "; ".join(unique_c5),
            'candidate_binary_labels': "; ".join(unique_cb),
            'strict_5class_qualified': "yes" if is_strict_5class else "no",
            'strict_5class_label': strict_5class_label,
            'majority_5class_qualified': "yes" if is_majority_5class else "no",
            'majority_5class_label': majority_5class_label,
            'majority_resolved_by_voting': "yes" if majority_resolved_by_voting else "no",
            'strict_binary_qualified': "yes" if is_strict_binary else "no",
            'strict_binary_label': strict_binary_label,
            'majority_binary_qualified': "yes" if is_majority_binary else "no",
            'majority_binary_label': majority_binary_label,
            'majority_binary_resolved_by_voting': "yes" if majority_binary_resolved_by_voting else "no",
            'conflict_status': conflict_status
        })
        
    df_groups = pd.DataFrame(hash_groups)
    df_groups.to_csv(os.path.join(AUDIT_DIR, "raw_only_hash_groups.csv"), index=False)
    
    # 5-class before vs after strict/majority distributions
    before_c5_counts = df_inv['candidate_5_class'].value_counts().to_dict()
    
    strict_5class_counts = df_groups[df_groups['strict_5class_qualified'] == 'yes']['strict_5class_label'].value_counts().to_dict()
    majority_5class_counts = df_groups[df_groups['majority_5class_qualified'] == 'yes']['majority_5class_label'].value_counts().to_dict()
    
    # Binary strict vs majority distributions
    strict_binary_counts = df_groups[df_groups['strict_binary_qualified'] == 'yes']['strict_binary_label'].value_counts().to_dict()
    majority_binary_counts = df_groups[df_groups['majority_binary_qualified'] == 'yes']['majority_binary_label'].value_counts().to_dict()
    
    # raw_only_label_summary.csv
    summary_rows = []
    all_possible_classes = ["Belly Pain", "Burping", "Discomfort", "Hungry", "Tired", "Exclude", "Unknown"]
    for cls in all_possible_classes:
        summary_rows.append({
            'candidate_5_class_label': cls,
            'raw_count_before_deduplication': before_c5_counts.get(cls, 0),
            'strict_count_after_deduplication': strict_5class_counts.get(cls, 0),
            'majority_count_after_deduplication': majority_5class_counts.get(cls, 0)
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(AUDIT_DIR, "raw_only_label_summary.csv"), index=False)
    
    # raw_only_candidate_dataset_summary.csv
    candidate_summary = [
        {
            'candidate_dataset': 'strict_5class',
            'total_samples': sum(strict_5class_counts.values()),
            'hungry_count': strict_5class_counts.get('Hungry', 0),
            'not_hungry_count': sum(strict_5class_counts.get(c, 0) for c in ["Belly Pain", "Burping", "Discomfort", "Tired"]),
            'belly_pain_count': strict_5class_counts.get('Belly Pain', 0),
            'burping_count': strict_5class_counts.get('Burping', 0),
            'discomfort_count': strict_5class_counts.get('Discomfort', 0),
            'tired_count': strict_5class_counts.get('Tired', 0)
        },
        {
            'candidate_dataset': 'majority_5class',
            'total_samples': sum(majority_5class_counts.values()),
            'hungry_count': majority_5class_counts.get('Hungry', 0),
            'not_hungry_count': sum(majority_5class_counts.get(c, 0) for c in ["Belly Pain", "Burping", "Discomfort", "Tired"]),
            'belly_pain_count': majority_5class_counts.get('Belly Pain', 0),
            'burping_count': majority_5class_counts.get('Burping', 0),
            'discomfort_count': majority_5class_counts.get('Discomfort', 0),
            'tired_count': majority_5class_counts.get('Tired', 0)
        },
        {
            'candidate_dataset': 'strict_binary',
            'total_samples': sum(strict_binary_counts.values()),
            'hungry_count': strict_binary_counts.get('Hungry', 0),
            'not_hungry_count': strict_binary_counts.get('Not Hungry', 0),
            'belly_pain_count': 0,
            'burping_count': 0,
            'discomfort_count': 0,
            'tired_count': 0
        },
        {
            'candidate_dataset': 'majority_binary',
            'total_samples': sum(majority_binary_counts.values()),
            'hungry_count': majority_binary_counts.get('Hungry', 0),
            'not_hungry_count': majority_binary_counts.get('Not Hungry', 0),
            'belly_pain_count': 0,
            'burping_count': 0,
            'discomfort_count': 0,
            'tired_count': 0
        }
    ]
    df_cand_summary = pd.DataFrame(candidate_summary)
    df_cand_summary.to_csv(os.path.join(AUDIT_DIR, "raw_only_candidate_dataset_summary.csv"), index=False)
    
    # Overlap metrics
    df_raw1 = df_inv[df_inv['source_root'] == 'raw_data']
    df_raw2 = df_inv[df_inv['source_root'] == 'raw_data_2']
    
    hashes_raw1 = set(df_raw1['md5_hash'].unique())
    hashes_raw2 = set(df_raw2['md5_hash'].unique())
    
    overlap_hashes = hashes_raw1 & hashes_raw2
    only_raw2_hashes = hashes_raw2 - hashes_raw1
    
    # Save Conflict Report
    with open(os.path.join(AUDIT_DIR, "raw_only_conflict_report.md"), "w", encoding="utf-8") as f:
        f.write("# Raw-Only Dataset Audit: Conflict Report\n\n")
        f.write(f"Total hash groups with label conflicts: **{len(conflict_report)}**\n\n")
        f.write("Identical audio files stored under contradictory directory labels within raw_data and raw_data_2:\n\n")
        for idx, item in enumerate(conflict_report):
            f.write(f"### Conflict Group {idx+1}: Hash `{item['md5_hash']}`\n")
            f.write(f"- **Number of copies:** {item['copies_count']}\n")
            f.write(f"- **Resolved Label (Majority Voting):** `{item['resolved_5_class']}`\n")
            f.write("- **Files and their assigned labels:**\n")
            for path, lbl in zip(item['file_paths'], item['c5_classes']):
                f.write(f"  - `{path}` -> **{lbl}**\n")
            f.write("\n" + "-"*50 + "\n\n")
            
    # Save Dataset Summary
    raw_labels = sorted(list(set(df_inv['inferred_raw_label_from_folder'].unique().tolist() + df_inv['inferred_raw_label_from_filename'].unique().tolist())))
    
    # Unknown labels
    unmapped_raw_labels = []
    for rl in raw_labels:
        l_low = rl.lower().strip()
        if l_low not in FIVE_CLASS_MAP and l_low not in EXCLUDE_LABELS and l_low != 'unknown':
            unmapped_raw_labels.append(rl)
            
    with open(os.path.join(AUDIT_DIR, "raw_only_dataset_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Raw-Only Dataset Audit: Summary Report\n\n")
        
        f.write("## 📋 Answers to Core Audit Questions\n\n")
        
        f.write("### 1. How many total audio files are in raw_data + raw_data_2 only?\n")
        f.write(f"**{len(df_inv)}** total audio files exist in the two raw folders (**{len(df_raw1)}** in `raw_data`, **{len(df_raw2)}** in `raw_data_2`).\n\n")
        
        f.write("### 2. How many unique MD5 hashes exist?\n")
        f.write(f"**{len(df_groups)}** unique physical assets exist.\n\n")
        
        f.write("### 3. How many duplicates exist?\n")
        f.write(f"**{len(df_inv) - len(df_groups)}** duplicate copies exist in raw_data + raw_data_2 (approx. **{(len(df_inv) - len(df_groups))/len(df_inv)*100:.2f}%** duplication rate).\n\n")
        
        f.write("### 4. How much overlap exists between raw_data and raw_data_2?\n")
        f.write(f"There is a massive overlap of **{len(overlap_hashes)}** unique physical files between the two directories. (i.e. **{len(overlap_hashes)/len(hashes_raw2)*100:.2f}%** of the unique files in `raw_data_2` also exist in `raw_data`).\n\n")
        
        f.write("### 5. How many files are truly new in raw_data_2 compared to raw_data?\n")
        f.write(f"Only **{len(only_raw2_hashes)}** unique physical files are truly new in `raw_data_2` compared to `raw_data`.\n\n")
        
        f.write("### 6. What are the raw labels found?\n")
        f.write("The raw labels include: " + ", ".join([f"`{rl}`" for rl in raw_labels]) + "\n\n")
        
        f.write("### 7. What is the 5-class distribution before deduplication?\n")
        f.write("| Class | Count |\n| :--- | :--- |\n")
        for cls in all_possible_classes:
            f.write(f"| {cls} | {before_c5_counts.get(cls, 0)} |\n")
        f.write("\n")
        
        f.write("### 8. What is the 5-class distribution after deduplication using strict mode?\n")
        f.write("| Class | Count |\n| :--- | :--- |\n")
        for cls in ["Belly Pain", "Burping", "Discomfort", "Hungry", "Tired"]:
            f.write(f"| {cls} | {strict_5class_counts.get(cls, 0)} |\n")
        f.write("\n")
        
        f.write("### 9. What is the 5-class distribution after deduplication using majority voting?\n")
        f.write("| Class | Count |\n| :--- | :--- |\n")
        for cls in ["Belly Pain", "Burping", "Discomfort", "Hungry", "Tired"]:
            f.write(f"| {cls} | {majority_5class_counts.get(cls, 0)} |\n")
        f.write("\n")
        
        f.write("### 10. What is the Hungry vs Not Hungry distribution in strict mode?\n")
        f.write("| Class | Count |\n| :--- | :--- |\n")
        f.write(f"| Hungry | {strict_binary_counts.get('Hungry', 0)} |\n")
        f.write(f"| Not Hungry | {strict_binary_counts.get('Not Hungry', 0)} |\n\n")
        
        f.write("### 11. What is the Hungry vs Not Hungry distribution in majority mode?\n")
        f.write("| Class | Count |\n| :--- | :--- |\n")
        f.write(f"| Hungry | {majority_binary_counts.get('Hungry', 0)} |\n")
        f.write(f"| Not Hungry | {majority_binary_counts.get('Not Hungry', 0)} |\n\n")
        
        f.write("### 12. How many MD5 groups have label conflicts?\n")
        f.write(f"**{len(conflict_report)}** unique hash groups have contradictory labeling.\n\n")
        
        f.write("### 13. Which folders or labels are unknown/unmapped?\n")
        if unmapped_raw_labels:
            f.write(", ".join([f"`{url}`" for url in unmapped_raw_labels]) + "\n\n")
        else:
            f.write("None\n\n")
            
        f.write("### 14. Which dataset candidate is recommended for training?\n")
        f.write("- **Recommendation for 5-Class Tasks:** `majority_5class` (total of **{0}** unique samples). Strict mode is overly conservative and excludes many valuable samples that are easily resolved via majority voting.\n".format(sum(majority_5class_counts.values())))
        f.write("- **Recommendation for Binary Tasks:** `majority_binary` (total of **{0}** unique samples). Independently resolving binary conflicts allows the preservation of more valid samples because some 5-class label conflicts (e.g. Belly Pain vs Discomfort) collapse into the single 'Not Hungry' class.\n".format(sum(majority_binary_counts.values())))
        
    print("Raw-only dataset audit completed successfully!")

if __name__ == "__main__":
    main()
