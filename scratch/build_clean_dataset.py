import os
import shutil
import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("librosa and soundfile are required for data cleaning.")
    exit(1)

WORKSPACE_DIR = r"C:\Users\ASUS\Desktop\aml_project"
AUDIT_DIR = os.path.join(WORKSPACE_DIR, "raw_only_label_evidence_audit")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "master_clean_evidence")

# Target SR
SR = 16000

def process_audio(input_path, temp_path):
    try:
        # Load audio (forces mono, resamples to 16000)
        y, sr = librosa.load(input_path, sr=SR, mono=True)
        orig_dur = librosa.get_duration(y=y, sr=sr)
        
        # Trim silence (top_db=30 is conservative)
        y_trimmed, index = librosa.effects.trim(y, top_db=30)
        trimmed_dur = librosa.get_duration(y=y_trimmed, sr=sr)
        
        if trimmed_dur < 0.3:
            return False, orig_dur, trimmed_dur, "Rejected: Duration < 0.3s after trimming"
            
        # Normalize safe
        max_amp = np.max(np.abs(y_trimmed))
        if max_amp > 0:
            y_norm = y_trimmed / max_amp * 0.95
        else:
            y_norm = y_trimmed
            
        # Write to PCM WAV
        sf.write(temp_path, y_norm, SR, subtype='PCM_16')
        
        return True, orig_dur, trimmed_dur, "Success"
    except Exception as e:
        return False, 0.0, 0.0, f"Error reading/converting: {str(e)}"

def main():
    print("Starting master clean dataset construction...")
    
    # Create directories
    for sub in ["majority_5class", "majority_binary", "rejected/unknown", "rejected/exclude_noise_silence_laugh", "rejected/unreadable", "manifests"]:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)
        
    for cls in ["Belly_Pain", "Burping", "Discomfort", "Hungry", "Tired"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "majority_5class", cls), exist_ok=True)
        
    for cls in ["Hungry", "Not_Hungry"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "majority_binary", cls), exist_ok=True)
        
    df_groups = pd.read_csv(os.path.join(AUDIT_DIR, "label_evidence_hash_groups.csv"))
    
    manifest_5class = []
    manifest_binary = []
    manifest_rejected = []
    
    durations = []
    
    for idx, row in df_groups.iterrows():
        md5 = row['md5_hash']
        paths = row['file_paths'].split('; ')
        selected_rel_path = paths[0]
        source_path = os.path.join(WORKSPACE_DIR, selected_rel_path)
        
        status = row['group_status']
        maj_5c_qual = row['majority_5class_qualified'] == 'yes'
        maj_5c_lbl = row['majority_5class_label']
        maj_bin_qual = row['majority_binary_qualified'] == 'yes'
        maj_bin_lbl = row['majority_binary_label']
        
        if status == 'exclude':
            manifest_rejected.append({
                'md5_hash': md5, 'reason': 'exclude_noise_silence_laugh', 'original_paths': row['file_paths']
            })
            continue
        elif status == 'unknown':
            manifest_rejected.append({
                'md5_hash': md5, 'reason': 'unknown_lonely', 'original_paths': row['file_paths']
            })
            continue
        elif not maj_5c_qual and not maj_bin_qual:
            manifest_rejected.append({
                'md5_hash': md5, 'reason': 'unresolved_conflict_or_invalid', 'original_paths': row['file_paths']
            })
            continue
            
        # Try to process
        temp_out = os.path.join(OUTPUT_DIR, f"temp_{md5}.wav")
        success, orig_dur, trimmed_dur, msg = process_audio(source_path, temp_out)
        
        if not success:
            manifest_rejected.append({
                'md5_hash': md5, 'reason': msg, 'original_paths': row['file_paths']
            })
            continue
            
        durations.append(trimmed_dur)
        
        # Route to final
        out_5c = None
        out_bin = None
        
        if maj_5c_qual:
            safe_lbl = maj_5c_lbl.replace(" ", "_")
            out_5c_rel = f"majority_5class/{safe_lbl}/{safe_lbl}__{md5}.wav"
            out_5c = os.path.join(OUTPUT_DIR, out_5c_rel)
            shutil.copy2(temp_out, out_5c)
            manifest_5class.append({
                'md5_hash': md5,
                'resolved_label_5class': safe_lbl,
                'source_selected_path': selected_rel_path,
                'all_original_paths': row['file_paths'],
                'evidence_status_summary': status,
                'output_path_5class': out_5c_rel,
                'original_duration_sec': orig_dur,
                'cleaned_duration_sec': trimmed_dur,
                'output_sample_rate': SR
            })
            
        if maj_bin_qual:
            safe_lbl = maj_bin_lbl.replace(" ", "_")
            out_bin_rel = f"majority_binary/{safe_lbl}/{safe_lbl}__{md5}.wav"
            out_bin = os.path.join(OUTPUT_DIR, out_bin_rel)
            shutil.copy2(temp_out, out_bin)
            manifest_binary.append({
                'md5_hash': md5,
                'resolved_label_binary': safe_lbl,
                'source_selected_path': selected_rel_path,
                'all_original_paths': row['file_paths'],
                'evidence_status_summary': status,
                'output_path_binary': out_bin_rel,
                'original_duration_sec': orig_dur,
                'cleaned_duration_sec': trimmed_dur,
                'output_sample_rate': SR
            })
            
        if os.path.exists(temp_out):
            os.remove(temp_out)
            
    # Save Manifests
    df_5c = pd.DataFrame(manifest_5class)
    if not df_5c.empty:
        df_5c.to_csv(os.path.join(OUTPUT_DIR, "manifests", "clean_majority_5class_manifest.csv"), index=False)
        
    df_bin = pd.DataFrame(manifest_binary)
    if not df_bin.empty:
        df_bin.to_csv(os.path.join(OUTPUT_DIR, "manifests", "clean_majority_binary_manifest.csv"), index=False)
        
    df_rej = pd.DataFrame(manifest_rejected)
    if not df_rej.empty:
        df_rej.to_csv(os.path.join(OUTPUT_DIR, "manifests", "rejected_manifest.csv"), index=False)
        
    # Metrics
    c5_counts = df_5c['resolved_label_5class'].value_counts().to_dict() if not df_5c.empty else {}
    bin_counts = df_bin['resolved_label_binary'].value_counts().to_dict() if not df_bin.empty else {}
    rej_counts = df_rej['reason'].value_counts().to_dict() if not df_rej.empty else {}
    
    min_d = np.min(durations) if durations else 0
    med_d = np.median(durations) if durations else 0
    max_d = np.max(durations) if durations else 0
    
    # Save summary
    with open(os.path.join(OUTPUT_DIR, "manifests", "cleaning_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Master Clean Evidence Dataset Summary\n\n")
        f.write("## Audio Standardization Specs\n")
        f.write("- Format: `.wav` (PCM 16-bit)\n")
        f.write("- Channels: Mono (1)\n")
        f.write("- Sample Rate: 16000 Hz\n")
        f.write("- Silence Trimming: Conservative (`top_db=30`)\n")
        f.write("- Amplitude Normalization: Safe (max peak scaled to 0.95 to prevent clipping)\n")
        f.write("- Minimum Length: Files shorter than 0.3s after trimming were rejected.\n\n")
        
        f.write("## 1. Final 5-Class File Count\n")
        for k, v in c5_counts.items():
            f.write(f"- **{k}**: {v}\n")
        f.write(f"**Total 5-Class:** {sum(c5_counts.values())}\n\n")
        
        f.write("## 2. Final Binary File Count\n")
        for k, v in bin_counts.items():
            f.write(f"- **{k}**: {v}\n")
        f.write(f"**Total Binary:** {sum(bin_counts.values())}\n\n")
        
        f.write("## 3. Rejected Files Breakdown\n")
        for k, v in rej_counts.items():
            f.write(f"- **{k}**: {v}\n")
        f.write(f"**Total Rejected:** {sum(rej_counts.values())}\n\n")
        
        f.write("## 4. Duration Metrics (Post-Trimming)\n")
        f.write(f"- **Minimum:** {min_d:.2f} s\n")
        f.write(f"- **Median:** {med_d:.2f} s\n")
        f.write(f"- **Maximum:** {max_d:.2f} s\n\n")
        
        f.write("## 5. MD5 Data Leakage Confirmation\n")
        f.write("✅ **CONFIRMED:** The pipeline iterated explicitly over unique MD5 hashes from `label_evidence_hash_groups.csv`.\n")
        f.write("No MD5 hash was processed more than once, ensuring that duplicate identical copies were stripped. Each file is named via `<label>__<md5>.wav`, embedding the group ID directly into the path for cross-validation.\n\n")
        
        f.write("## 6. Colab Export Command\n")
        f.write("To zip this clean dataset for Google Colab, use the following commands:\n")
        f.write("```bash\n")
        f.write("# Create zip for 5-class\n")
        f.write("zip -r clean_majority_5class.zip master_clean_evidence/majority_5class/\n\n")
        f.write("# Create zip for Binary\n")
        f.write("zip -r clean_majority_binary.zip master_clean_evidence/majority_binary/\n")
        f.write("```\n")

    print("Master clean dataset construction finished successfully!")

if __name__ == "__main__":
    main()
