# Raw-Only Dataset Audit: Summary Report

## 📋 Answers to Core Audit Questions

### 1. How many total audio files are in raw_data + raw_data_2 only?
**7651** total audio files exist in the two raw folders (**3554** in `raw_data`, **4097** in `raw_data_2`).

### 2. How many unique MD5 hashes exist?
**1052** unique physical assets exist.

### 3. How many duplicates exist?
**6599** duplicate copies exist in raw_data + raw_data_2 (approx. **86.25%** duplication rate).

### 4. How much overlap exists between raw_data and raw_data_2?
There is a massive overlap of **1002** unique physical files between the two directories. (i.e. **99.40%** of the unique files in `raw_data_2` also exist in `raw_data`).

### 5. How many files are truly new in raw_data_2 compared to raw_data?
Only **6** unique physical files are truly new in `raw_data_2` compared to `raw_data`.

### 6. What are the raw labels found?
The raw labels include: `Hungry`, `Tired`, `Uncomfortable`, `belly pain`, `belly_pain`, `burping`, `cold_hot`, `cry`, `discomfort`, `hungry`, `laugh`, `lonely`, `noise`, `not_cry`, `scared`, `silence`, `tired`, `unknown`

### 7. What is the 5-class distribution before deduplication?
| Class | Count |
| :--- | :--- |
| Belly Pain | 531 |
| Burping | 395 |
| Discomfort | 998 |
| Hungry | 2350 |
| Tired | 524 |
| Exclude | 1620 |
| Unknown | 1233 |

### 8. What is the 5-class distribution after deduplication using strict mode?
| Class | Count |
| :--- | :--- |
| Belly Pain | 48 |
| Burping | 39 |
| Discomfort | 69 |
| Hungry | 30 |
| Tired | 20 |

### 9. What is the 5-class distribution after deduplication using majority voting?
| Class | Count |
| :--- | :--- |
| Belly Pain | 63 |
| Burping | 46 |
| Discomfort | 85 |
| Hungry | 410 |
| Tired | 39 |

### 10. What is the Hungry vs Not Hungry distribution in strict mode?
| Class | Count |
| :--- | :--- |
| Hungry | 30 |
| Not Hungry | 232 |

### 11. What is the Hungry vs Not Hungry distribution in majority mode?
| Class | Count |
| :--- | :--- |
| Hungry | 410 |
| Not Hungry | 233 |

### 12. How many MD5 groups have label conflicts?
**437** unique hash groups have contradictory labeling.

### 13. Which folders or labels are unknown/unmapped?
`cry`, `lonely`, `scared`

### 14. Which dataset candidate is recommended for training?
- **Recommendation for 5-Class Tasks:** `majority_5class` (total of **643** unique samples). Strict mode is overly conservative and excludes many valuable samples that are easily resolved via majority voting.
- **Recommendation for Binary Tasks:** `majority_binary` (total of **643** unique samples). Independently resolving binary conflicts allows the preservation of more valid samples because some 5-class label conflicts (e.g. Belly Pain vs Discomfort) collapse into the single 'Not Hungry' class.
