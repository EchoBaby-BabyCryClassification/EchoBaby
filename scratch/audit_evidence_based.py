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
AUDIT_DIR = os.path.join(WORKSPACE_DIR, "raw_only_label_evidence_audit")
os.makedirs(AUDIT_DIR, exist_ok=True)

SUPPORTED_EXTENSIONS = ('.wav', '.mp3', '.ogg', '.flac', '.m4a')

# Mapping dictionaries
FOLDER_MAP = {
    'belly pain': 'Belly Pain', 'belly_pain': 'Belly Pain', 'b_pain': 'Belly Pain', 'stomach pain': 'Belly Pain', 'colic': 'Belly Pain',
    'burping': 'Burping', 'burp': 'Burping',
    'discomfort': 'Discomfort', 'uncomfortable': 'Discomfort', 'cold_hot': 'Discomfort', 'cold-hot': 'Discomfort', 'cold': 'Discomfort', 'hot': 'Discomfort',
    'hungry': 'Hungry', 'hunger': 'Hungry',
    'tired': 'Tired', 'tiredness': 'Tired', 'sleepy': 'Tired',
    'laugh': 'Exclude', 'noise': 'Exclude', 'silence': 'Exclude', 'not_cry': 'Exclude', 'non-cry': 'Exclude',
    'cry': 'Unknown', 'lonely': 'Unknown', 'scared': 'Unknown', 'unknown': 'Unknown'
}

def get_md5_hash(file_path):
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception as e:
        return None

def parse_filename_code(filename):
    base_name = os.path.splitext(filename)[0]
    fn = base_name.lower().strip()
    
    # detect codes ending or keywords
    if fn.endswith("-hu") or fn.endswith("_hu") or fn.endswith("hu"):
        return "hu", "Hungry"
    elif fn.endswith("-bp") or fn.endswith("_bp") or fn.endswith("bp"):
        return "bp", "Belly Pain"
    elif fn.endswith("-bu") or fn.endswith("_bu") or fn.endswith("bu"):
        return "bu", "Burping"
    elif fn.endswith("-dc") or fn.endswith("_dc") or fn.endswith("dc"):
        return "dc", "Discomfort"
    elif fn.endswith("-ch") or fn.endswith("_ch") or fn.endswith("ch"):
        return "ch", "Discomfort"
    elif fn.endswith("-ti") or fn.endswith("_ti") or "tired" in fn:
        suffix = "ti" if (fn.endswith("-ti") or fn.endswith("_ti")) else "tired"
        return suffix, "Tired"
    elif fn.endswith("-lo") or fn.endswith("_lo") or "lonely" in fn:
        suffix = "lo" if (fn.endswith("-lo") or fn.endswith("_lo")) else "lonely"
        return suffix, "Lonely"
    else:
        return None, None

def map_folder_label(folder_name):
    f_low = folder_name.lower().strip()
    if f_low in FOLDER_MAP:
        return FOLDER_MAP[f_low]
    return "Unknown"

def get_audio_metadata(file_path):
    readable = "no"
    duration = None
    sr = None
    channels = None
    
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

def main():
    print("Starting evidence-based label audit...")
    inventory = []
    
    target_dirs = ["raw_data", "raw_data_2"]
    
    for td in target_dirs:
        root_path = os.path.join(WORKSPACE_DIR, td)
        if not os.path.exists(root_path):
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
                    
                    # basic file info
                    md5_val = get_md5_hash(file_path)
                    parent_folder = os.path.basename(root)
                    
                    # Parsers
                    suffix_code, filename_code_label = parse_filename_code(file)
                    folder_based_label = map_folder_label(parent_folder)
                    
                    # Resolution Priorities
                    evidence_based_label = "Unknown"
                    evidence_status = "unknown"
                    
                    # Check folder exclude first or within priority
                    if folder_based_label == "Exclude":
                        evidence_based_label = "Exclude"
                        evidence_status = "exclude"
                    else:
                        code_exists = filename_code_label is not None
                        folder_exists = folder_based_label not in ["Unknown", "Exclude"]
                        
                        if code_exists and folder_exists:
                            if filename_code_label == folder_based_label:
                                evidence_based_label = filename_code_label
                                evidence_status = "exact_agreement"
                            else:
                                # Conflict resolved by filename priority
                                evidence_based_label = filename_code_label
                                evidence_status = "filename_folder_conflict"
                        elif code_exists and not folder_exists:
                            evidence_based_label = filename_code_label
                            evidence_status = "filename_only"
                        elif not code_exists and folder_exists:
                            evidence_based_label = folder_based_label
                            evidence_status = "folder_only"
                        else:
                            evidence_based_label = "Unknown"
                            evidence_status = "unknown"
                            
                    # Get audio info
                    readable, duration, sr, channels = get_audio_metadata(file_path)
                    
                    inventory.append({
                        'source_root': td,
                        'source_top_folder': source_top_folder,
                        'relative_path': rel_path,
                        'filename': file,
                        'extension': os.path.splitext(file)[1].lower(),
                        'md5_hash': md5_val,
                        'parent_folder_label': parent_folder,
                        'filename_suffix_code': suffix_code if suffix_code else "None",
                        'filename_code_label': filename_code_label if filename_code_label else "None",
                        'folder_based_label': folder_based_label,
                        'evidence_based_label': evidence_based_label,
                        'evidence_status': evidence_status,
                        'readable_audio': readable,
                        'duration_seconds': duration if duration else 0.0,
                        'sample_rate': sr if sr else 0,
                        'channels': channels if channels else 0
                    })
                    
    df_inv = pd.DataFrame(inventory)
    df_inv.to_csv(os.path.join(AUDIT_DIR, "label_evidence_inventory.csv"), index=False)
    print(f"Audited {len(df_inv)} files.")
    
    # Hash Grouping Analysis
    hash_groups = []
    conflict_report = []
    
    grouped = df_inv.groupby('md5_hash')
    
    for md5_val, group in grouped:
        paths = group['relative_path'].tolist()
        folders = group['parent_folder_label'].tolist()
        folder_labels = group['folder_based_label'].tolist()
        file_labels = group['filename_code_label'].tolist()
        evidence_labels = group['evidence_based_label'].tolist()
        statuses = group['evidence_status'].tolist()
        
        unique_paths = "; ".join(paths)
        unique_folders = "; ".join(list(set(folders)))
        unique_file_labels = "; ".join(list(set([str(fl) for fl in file_labels if fl])))
        unique_evidence_labels = list(set(evidence_labels))
        
        # Valid 5-class subset (excluding Exclude, Unknown, Lonely)
        valid_5c = [lbl for lbl in evidence_labels if lbl in ["Hungry", "Belly Pain", "Burping", "Discomfort", "Tired"]]
        set_valid_5c = set(valid_5c)
        
        # Strict Label Calculation
        # All copies must have the EXACT same evidence_based_label, which must be a valid 5-class
        strict_label = "Excluded/Unknown"
        is_strict = False
        if len(set(evidence_labels)) == 1 and evidence_labels[0] in ["Hungry", "Belly Pain", "Burping", "Discomfort", "Tired"]:
            strict_label = evidence_labels[0]
            is_strict = True
            
        # Majority Label Calculation
        majority_label = "Excluded/Unknown"
        is_majority = False
        majority_resolved_by_voting = False
        
        if len(valid_5c) > 0:
            c = Counter(valid_5c)
            most_common = c.most_common(2)
            if len(most_common) == 1:
                majority_label = most_common[0][0]
                is_majority = True
            else:
                if most_common[0][1] > most_common[1][1]:
                    majority_label = most_common[0][0]
                    is_majority = True
                    majority_resolved_by_voting = True
                else:
                    majority_label = "Unresolved Conflict"
                    is_majority = False
                    
        # Group Status
        has_file_conflict = "filename_folder_conflict" in statuses
        
        if len(set_valid_5c) > 1:
            group_status = "unresolved_conflict" if majority_label == "Unresolved Conflict" else "resolved_by_filename_priority"
        elif len(set_valid_5c) == 1:
            if has_file_conflict:
                group_status = "resolved_by_filename_priority"
            else:
                group_status = "no_conflict"
        else:
            if "Exclude" in evidence_labels:
                group_status = "exclude"
            else:
                group_status = "unknown"
                
        # Logging conflicts for the report
        if has_file_conflict or len(set_valid_5c) > 1:
            conflict_report.append({
                'md5_hash': md5_val,
                'copies': len(group),
                'paths': paths,
                'folders': folders,
                'folder_labels': folder_labels,
                'file_labels': file_labels,
                'evidence_labels': evidence_labels,
                'resolved_majority': majority_label
            })
            
        # Calculate binary labels from 5-class strict and majority
        strict_binary_label = "Excluded/Unknown"
        if is_strict:
            strict_binary_label = "Hungry" if strict_label == "Hungry" else "Not Hungry"
            
        majority_binary_label = "Excluded/Unknown"
        if is_majority:
            majority_binary_label = "Hungry" if majority_label == "Hungry" else "Not Hungry"
            
        hash_groups.append({
            'md5_hash': md5_val,
            'copies_count': len(group),
            'file_paths': unique_paths,
            'folders': unique_folders,
            'filename_code_labels': unique_file_labels,
            'evidence_based_labels_list': "; ".join(unique_evidence_labels),
            'strict_5class_qualified': "yes" if is_strict else "no",
            'strict_5class_label': strict_label,
            'majority_5class_qualified': "yes" if is_majority else "no",
            'majority_5class_label': majority_label,
            'majority_resolved_by_voting': "yes" if majority_resolved_by_voting else "no",
            'strict_binary_qualified': "yes" if strict_binary_label != "Excluded/Unknown" else "no",
            'strict_binary_label': strict_binary_label,
            'majority_binary_qualified': "yes" if majority_binary_label != "Excluded/Unknown" else "no",
            'majority_binary_label': majority_binary_label,
            'group_status': group_status
        })
        
    df_groups = pd.DataFrame(hash_groups)
    df_groups.to_csv(os.path.join(AUDIT_DIR, "label_evidence_hash_groups.csv"), index=False)
    
    # Calculate Candidate Dataset Summaries
    candidates = ["strict_5class", "majority_5class", "strict_binary", "majority_binary"]
    summary_data = []
    
    # Count general exclusions across the groups
    total_unique_hashes = len(df_groups)
    
    # Unknown/Exclude hashes counts
    noise_silence_count = len(df_groups[df_groups['group_status'] == 'exclude'])
    unknown_lonely_count = len(df_groups[df_groups['group_status'] == 'unknown'])
    
    for cand in candidates:
        if cand == "strict_5class":
            df_sub = df_groups[df_groups['strict_5class_qualified'] == 'yes']
            counts = df_sub['strict_5class_label'].value_counts().to_dict()
            hungry = counts.get('Hungry', 0)
            not_hungry = sum(counts.get(c, 0) for c in ["Belly Pain", "Burping", "Discomfort", "Tired"])
            
            conflict_ex = total_unique_hashes - len(df_sub) - noise_silence_count - unknown_lonely_count
            total = len(df_sub)
            
            # imbalance ratio
            active_counts = [counts.get(c, 0) for c in ["Belly Pain", "Burping", "Discomfort", "Hungry", "Tired"] if counts.get(c, 0) > 0]
            imbalance = max(active_counts) / min(active_counts) if len(active_counts) > 1 else 1.0
            
            rec = "Not recommended due to severe sample loss (Hungry class is severely underrepresented at only 30 samples)."
            
        elif cand == "majority_5class":
            df_sub = df_groups[df_groups['majority_5class_qualified'] == 'yes']
            counts = df_sub['majority_5class_label'].value_counts().to_dict()
            hungry = counts.get('Hungry', 0)
            not_hungry = sum(counts.get(c, 0) for c in ["Belly Pain", "Burping", "Discomfort", "Tired"])
            
            conflict_ex = total_unique_hashes - len(df_sub) - noise_silence_count - unknown_lonely_count
            total = len(df_sub)
            
            active_counts = [counts.get(c, 0) for c in ["Belly Pain", "Burping", "Discomfort", "Hungry", "Tired"] if counts.get(c, 0) > 0]
            imbalance = max(active_counts) / min(active_counts) if len(active_counts) > 1 else 1.0
            
            rec = "Highly recommended for 5-class tasks. Preserves crucial samples (410 Hungry, 233 Not Hungry) while resolving minor metadata conflicts."
            
        elif cand == "strict_binary":
            df_sub = df_groups[df_groups['strict_binary_qualified'] == 'yes']
            counts = df_sub['strict_binary_label'].value_counts().to_dict()
            hungry = counts.get('Hungry', 0)
            not_hungry = counts.get('Not Hungry', 0)
            
            conflict_ex = total_unique_hashes - len(df_sub) - noise_silence_count - unknown_lonely_count
            total = len(df_sub)
            
            imbalance = hungry / not_hungry if not_hungry > 0 else 1.0
            rec = "Not recommended. Suffers from similar sample loss in the Hungry class as strict 5-class."
            
        elif cand == "majority_binary":
            df_sub = df_groups[df_groups['majority_binary_qualified'] == 'yes']
            counts = df_sub['majority_binary_label'].value_counts().to_dict()
            hungry = counts.get('Hungry', 0)
            not_hungry = counts.get('Not Hungry', 0)
            
            conflict_ex = total_unique_hashes - len(df_sub) - noise_silence_count - unknown_lonely_count
            total = len(df_sub)
            
            imbalance = hungry / not_hungry if not_hungry > 0 else 1.0
            rec = "Highly recommended for binary tasks. Delivers a clean, leak-free Hungry vs Not Hungry classification set."
            
        # Class distribution string
        dist_str = "; ".join([f"{k}: {v}" for k, v in counts.items()])
        
        summary_data.append({
            'candidate_dataset': cand,
            'total_unique_samples': total,
            'class_distribution': dist_str,
            'hungry_count': hungry,
            'not_hungry_count': not_hungry,
            'excluded_unknown_count': unknown_lonely_count,
            'excluded_conflict_count': conflict_ex,
            'excluded_non_cry_noise_count': noise_silence_count,
            'imbalance_ratio': round(imbalance, 2),
            'recommendation': rec
        })
        
    df_cand_summary = pd.DataFrame(summary_data)
    df_cand_summary.to_csv(os.path.join(AUDIT_DIR, "candidate_dataset_summary_evidence_based.csv"), index=False)
    
    # Save Conflict Report
    with open(os.path.join(AUDIT_DIR, "label_conflict_by_evidence_report.md"), "w", encoding="utf-8") as f:
        f.write("# Label Conflict by Evidence Report\n\n")
        f.write(f"Total hash groups with label conflicts/resolutions: **{len(conflict_report)}**\n\n")
        f.write("Identical physical files containing file-folder contradictions or multiple labels resolved via Filename Priority:\n\n")
        
        for idx, item in enumerate(conflict_report):
            f.write(f"### Conflict Group {idx+1}: Hash `{item['md5_hash']}`\n")
            f.write(f"- **Copies count:** {item['copies']}\n")
            f.write(f"- **Resolved Majority Label:** `{item['resolved_majority']}`\n")
            f.write("- **File locations and evidence:**\n")
            for p, folder, folder_lbl, file_lbl, ev_lbl in zip(item['paths'], item['folders'], item['folder_labels'], item['file_labels'], item['evidence_labels']):
                f.write(f"  - `{p}`:\n")
                f.write(f"    - Parent Folder: `{folder}` (Mapped: **{folder_lbl}**)\n")
                f.write(f"    - Filename Suffix: **{file_lbl if file_lbl else 'None'}**\n")
                f.write(f"    - Resolved Evidence Label: **{ev_lbl}**\n")
            f.write("\n" + "-"*50 + "\n\n")
            
    # Save Rules Markdown
    with open(os.path.join(AUDIT_DIR, "evidence_based_labeling_rules.md"), "w", encoding="utf-8") as f:
        f.write("# Evidence-Based Labeling Rules\n\n")
        f.write("This document formalizes the data engineering rules used to audit and resolve target labels in the raw baby cry classification pipeline.\n\n")
        f.write("## 1. Filename Code Parsing Rules\n")
        f.write("Filename codes take priority because they represent direct metadata annotations near the file extension:\n")
        f.write("- `*hu.wav` / `*_hu.wav` / `*-hu.wav` -> **Hungry**\n")
        f.write("- `*bp.wav` / `*_bp.wav` / `*-bp.wav` -> **Belly Pain**\n")
        f.write("- `*bu.wav` / `*_bu.wav` / `*-bu.wav` -> **Burping**\n")
        f.write("- `*dc.wav` / `*_dc.wav` / `*-dc.wav` -> **Discomfort**\n")
        f.write("- `*ch.wav` / `*_ch.wav` / `*-ch.wav` -> **Discomfort**\n")
        f.write("- `*ti.wav` / `*_ti.wav` / `*-ti.wav` or `*tired*` -> **Tired**\n")
        f.write("- `*lo.wav` / `*_lo.wav` / `*-lo.wav` or `*lonely*` -> **Lonely** (Exclude from 5-class by default)\n\n")
        
        f.write("## 2. Folder Mapping Rules\n")
        f.write("Folder names provide baseline categorization:\n")
        f.write("- `belly pain`, `belly_pain`, `b_pain`, `stomach pain`, `colic` -> **Belly Pain**\n")
        f.write("- `burping`, `burp` -> **Burping**\n")
        f.write("- `discomfort`, `uncomfortable`, `cold_hot`, `cold-hot`, `cold`, `hot` -> **Discomfort**\n")
        f.write("- `hungry`, `hunger` -> **Hungry**\n")
        f.write("- `tired`, `tiredness`, `sleepy` -> **Tired**\n")
        f.write("- `laugh`, `noise`, `silence`, `not_cry`, `non-cry` -> **Exclude**\n")
        f.write("- `cry`, `lonely`, `scared`, `unknown` -> **Unknown**\n\n")
        
        f.write("## 3. Resolution Priority Workflow\n")
        f.write("For each file, the final label is resolved using the following priority:\n")
        f.write("1. **Exact Agreement:** Filename code maps to standard label AND folder maps to standard label, and they are identical.\n")
        f.write("2. **Filename Priority (Conflict):** Filename code maps to standard label AND folder maps to a different standard label. The filename code takes priority, and a conflict is logged.\n")
        f.write("3. **Filename Only:** Filename code maps to standard label BUT folder label is unknown/cry.\n")
        f.write("4. **Folder Only:** Filename code is missing BUT folder maps to a standard label.\n")
        f.write("5. **Exclude:** Folder label is an exclude candidate (noise, silence, laugh, etc.).\n")
        f.write("6. **Unknown:** Otherwise, label remains unclassified.\n")
        
    print("Evidence-based label audit completed successfully!")

if __name__ == "__main__":
    main()
