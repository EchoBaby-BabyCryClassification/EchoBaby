# Master Clean Evidence Dataset Summary

## Audio Standardization Specs
- Format: `.wav` (PCM 16-bit)
- Channels: Mono (1)
- Sample Rate: 16000 Hz
- Silence Trimming: Conservative (`top_db=30`)
- Amplitude Normalization: Safe (max peak scaled to 0.95 to prevent clipping)
- Minimum Length: Files shorter than 0.3s after trimming were rejected.

## 1. Final 5-Class File Count
- **Hungry**: 410
- **Discomfort**: 82
- **Belly_Pain**: 63
- **Burping**: 46
- **Tired**: 38
**Total 5-Class:** 639

## 2. Final Binary File Count
- **Hungry**: 410
- **Not_Hungry**: 229
**Total Binary:** 639

## 3. Rejected Files Breakdown
- **exclude_noise_silence_laugh**: 324
- **unknown_lonely**: 86
- **Error reading/converting: **: 3
**Total Rejected:** 413

## 4. Duration Metrics (Post-Trimming)
- **Minimum:** 1.63 s
- **Median:** 6.40 s
- **Maximum:** 7.73 s

## 5. MD5 Data Leakage Confirmation
✅ **CONFIRMED:** The pipeline iterated explicitly over unique MD5 hashes from `label_evidence_hash_groups.csv`.
No MD5 hash was processed more than once, ensuring that duplicate identical copies were stripped. Each file is named via `<label>__<md5>.wav`, embedding the group ID directly into the path for cross-validation.

## 6. Colab Export Command
To zip this clean dataset for Google Colab, use the following commands:
```bash
# Create zip for 5-class
zip -r clean_majority_5class.zip master_clean_evidence/majority_5class/

# Create zip for Binary
zip -r clean_majority_binary.zip master_clean_evidence/majority_binary/
```
