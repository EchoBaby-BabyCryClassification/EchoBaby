# Evidence-Based Labeling Rules

This document formalizes the data engineering rules used to audit and resolve target labels in the raw baby cry classification pipeline.

## 1. Filename Code Parsing Rules
Filename codes take priority because they represent direct metadata annotations near the file extension:
- `*hu.wav` / `*_hu.wav` / `*-hu.wav` -> **Hungry**
- `*bp.wav` / `*_bp.wav` / `*-bp.wav` -> **Belly Pain**
- `*bu.wav` / `*_bu.wav` / `*-bu.wav` -> **Burping**
- `*dc.wav` / `*_dc.wav` / `*-dc.wav` -> **Discomfort**
- `*ch.wav` / `*_ch.wav` / `*-ch.wav` -> **Discomfort**
- `*ti.wav` / `*_ti.wav` / `*-ti.wav` or `*tired*` -> **Tired**
- `*lo.wav` / `*_lo.wav` / `*-lo.wav` or `*lonely*` -> **Lonely** (Exclude from 5-class by default)

## 2. Folder Mapping Rules
Folder names provide baseline categorization:
- `belly pain`, `belly_pain`, `b_pain`, `stomach pain`, `colic` -> **Belly Pain**
- `burping`, `burp` -> **Burping**
- `discomfort`, `uncomfortable`, `cold_hot`, `cold-hot`, `cold`, `hot` -> **Discomfort**
- `hungry`, `hunger` -> **Hungry**
- `tired`, `tiredness`, `sleepy` -> **Tired**
- `laugh`, `noise`, `silence`, `not_cry`, `non-cry` -> **Exclude**
- `cry`, `lonely`, `scared`, `unknown` -> **Unknown**

## 3. Resolution Priority Workflow
For each file, the final label is resolved using the following priority:
1. **Exact Agreement:** Filename code maps to standard label AND folder maps to standard label, and they are identical.
2. **Filename Priority (Conflict):** Filename code maps to standard label AND folder maps to a different standard label. The filename code takes priority, and a conflict is logged.
3. **Filename Only:** Filename code maps to standard label BUT folder label is unknown/cry.
4. **Folder Only:** Filename code is missing BUT folder maps to a standard label.
5. **Exclude:** Folder label is an exclude candidate (noise, silence, laugh, etc.).
6. **Unknown:** Otherwise, label remains unclassified.
