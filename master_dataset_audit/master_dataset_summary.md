# Master Dataset Audit: Summary Report

## Executive Summary
This report presents the master audit and data engineering inventory of the baby cry classification project before any model training or dataset cleanup. The entire project repository was scanned to establish a robust, reproducible, and leak-free foundation.

## 📋 Answers to Key Audit Questions

### 1. How many total audio files exist?
**9721** total audio files were found across all directories.

### 2. How many unique MD5 hashes exist?
**3107** unique MD5 hashes exist, representing the true number of physical audio assets.

### 3. How many internal duplicates exist?
**6614** internal duplicates exist in the repository (approx. **68.04%** duplication rate).

### 4. Which source folders contain real audio?
The following folders contain valid, readable audio files:
- `cleaned_audio_unique/` (634 files, 634 readable)
- `raw_data/` (3554 files, 3551 readable)
- `raw_data_2/` (4097 files, 4094 readable)
- `standardized_audio/` (1436 files, 1436 readable)

### 5. Which source folders contain only metadata or no audio?
Folders scanned that did not yield direct readable audio files (or contain only non-audio assets):
- None (all scanned directories containing audio files yielded readable audio assets).

### 6. What raw labels were found?
The raw labels inferred from parent folder names or filenames include:
`Hungry`, `Tired`, `Uncomfortable`, `belly pain`, `belly_pain`, `burping`, `cleaned_audio_unique`, `cold_hot`, `cry`, `discomfort`, `hungry`, `laugh`, `lonely`, `noise`, `not_cry`, `scared`, `silence`, `standardized_audio`, `tired`, `unknown`

### 7. What is the candidate 5-class distribution before deduplication?
| Candidate 5-Class Label | Raw Count (Before Deduplication) |
| :--- | :--- |
| Belly Pain | 531 |
| Burping | 395 |
| Discomfort | 998 |
| Hungry | 2350 |
| Tired | 524 |
| Exclude | 1836 |
| Unknown | 3087 |

### 8. What is the candidate 5-class distribution after deduplication?
| Candidate 5-Class Label | Unique Count (After Deduplication) |
| :--- | :--- |
| Belly Pain | 63 |
| Burping | 46 |
| Discomfort | 85 |
| Hungry | 410 |
| Tired | 39 |
| Exclude | 540 |
| Unknown | 1924 |

### 9. What is the candidate Hungry vs Not Hungry distribution after deduplication?
| Candidate Binary Label | Count (After Deduplication) |
| :--- | :--- |
| Hungry | 410 |
| Not Hungry | 233 |
| Exclude | 540 |
| Unknown | 1924 |

### 10. How many hash groups have label conflicts?
**437** hash groups exhibit label conflicts (identical audio files stored under different directories with contradictory raw labels). These are detailed in [master_conflict_report.md](master_conflict_report.md).

### 11. Which labels are unknown or unmapped?
The following raw labels did not match any category in the candidate mapping rules:
`cleaned_audio_unique`, `cry`, `lonely`, `scared`, `standardized_audio`

### 12. What is the recommended next step?
1. **Programmatic Deduplication:** Run a clean rebuild script to copy only unique physical audio files (using resolved majority labels) to a new, standardized directory.
2. **Group Split Preservation:** Use the MD5 hash group as the `Group ID` in all subsequent `StratifiedGroupKFold` splittings to mathematically guarantee no data leakage.
3. **Pediatric Specialist Review:** Target the **Belly Pain** and **Discomfort** classes (which represent the main error patterns and highest conflict areas) for manual audio audits by a medical professional.
