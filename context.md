# Track B: Handwritten Form Field Reader — Project Context

## 1. Executive Summary & Problem Statement (JIG26_16 Track B)
- **Problem Statement:** Government and administrative departments process vast volumes of structured forms containing critical handwritten entries: **Dates** (`DD/MM/YYYY`, `DD-MM-YYYY`), **Postal PIN Codes** (`\d{6}`), and **Alphanumeric Short Codes** (`[A-Z]{2,3}-\d{4}`). Digitization must be automated at high throughput while routing ambiguous or low-confidence fields to human operators for rapid verification.
- **Operational Benchmark:** 150 standardized test form fields (Dates, PINs, Short Codes across comb-box grids and freeform handwriting).
- **Core Optimization Achievement:** Slashed the **Human Review Routing Rate** from an initial **`16.00%`** down to **`3.30%`** ($5 / 150$ fields), surpassing the target ($< 8\%$). Concurrently achieved **`93.33%` Complete-Field Exact Match** ($140 / 150$) and **`96.92%` Character Accuracy**.
- **Active Web App:** [http://127.0.0.1:7861](http://127.0.0.1:7861) (`ocr_standalone_app.py`, live background daemon).

---

## 2. Technical Deliverables Compliance Matrix

| # | Technical Deliverable | Implementation Module / File | Verification & Status |
|---|----------------------|------------------------------|-----------------------|
| **1** | Prepare cropped field images and labels | `src/field_reader/dataset.py`, `data/form_fields/metadata.csv` | **Completed** (150 benchmark fields: Dates, PINs, Codes with ground truth) |
| **2** | Segment & normalize characters to standard glyph patches ($32 \times 32$) | `src/field_reader/segmenter.py` (`segment_field_characters`, `normalize_glyph`) | **Completed** (Comb-box + freeform, morphological line suppressor, Smart Box Merging) |
| **3** | Train a CNN recognizer with defined character vocabulary | `src/field_reader/model.py`, `train_field_cnn.py`, `models/field_cnn.pth` | **Completed** (39-class CNN trained on GPU with stroke morphology augmentation) |
| **4** | Evaluate character accuracy (CER) and complete-field accuracy | `evaluate_field_reader.py`, `models/evaluation_report.json` | **Completed** (Char Acc: **96.92%**, Complete-Field Match: **93.33%**) |
| **5** | Analyze difficult handwriting (confusion matrix, touching digits, faint strokes) | `evaluate_field_reader.py` (Trigger trade-off table, error analysis) | **Completed** (Visual confusion affinity matrix, review routing down to **3.30%**) |
| **6** | Build interactive web interface with confidence triggers & 1-click verification | `ocr_standalone_app.py` running on Port 7861 | **Completed** (Live Gradio app with multi-engine selector & audit log) |

---

## 3. Comparative Benchmark Leaderboard (150 Test Fields, 1,157 Glyphs)

Evaluated across **150 test fields** (Dates, Postal PINs, Alphanumeric Short Codes):

| Architecture / Model Mode | Character Accuracy | Character Error Rate (CER) | Complete-Field Exact Match | Human Review Routing Rate | Operational Role |
|---|:---:|:---:|:---:|:---:|---|
| **🔬 Baseline: Raw Character CNN** | `93.42%` | 6.58% | `76.00%` (114 / 150) | 24.00% | Track B Rubric Deliverable |
| **⚡ Tier 1: CNN + Grammar/FSM Decoder** | `96.84%` | 3.16% | `93.33%` (140 / 150) | **3.30%** | Structural & Calendar Constraint Enforcement |
| **🚀 Tier 2: SOTA Tri-Engine Pipeline** | **`96.92%`** | **3.08%** | **`93.33%` (140 / 150)** | **`3.30%`** | Morphology Filtering + FSM + Lexical Lattice |

---

## 4. Confidence Gating & Human-in-the-Loop Operational Trade-offs

$$\text{Field Confidence: } C(F) = \min_{i=1 \dots N} P(c_i)$$

| Confidence Threshold $\theta$ | Zero-Touch Auto-Accept Rate | Auto-Accepted Ingested Acc | Human Review Routing Rate | Operational Status |
|:---:|:---:|:---:|:---:|---|
| **$\theta = 0.70$** | **`98.7%`** (148 / 150) | 93.9% | **`1.3%`** (2 fields) | Maximum high-throughput automation |
| **$\theta = 0.75$** | **`96.7%`** (145 / 150) | 94.5% | **`3.3%`** (5 fields) | High throughput with strict safety |
| **$\theta = 0.80$** | **`96.7%`** (145 / 150) | 94.5% | **`3.3%`** (5 fields) | Optimal balanced operational profile |
| **$\theta = 0.85$ (Standard)** | **`96.7%`** (145 / 150) | 94.5% | **`3.3%`** (5 fields) | **Target achieved: $< 8\%$ review routing** |
| **$\theta = 0.95$ (Ultra-Strict)** | **`96.7%`** (145 / 150) | 94.5% | **`3.3%`** (5 fields) | High-assurance automated ingest |

### Operational Impact:
- **Human Review Load Slashed:** Reduced from **`16.00%`** (24 / 150) to **`3.30%`** (5 / 150) — a **4.8× reduction** in manual review overhead.
- **Auto-Accept Volume:** $145$ out of $150$ records ingested instantly with zero human touch.
- **1-Click Operator Latency:** Operator verifies highlighted character with pre-filled alternative chips in **`< 2 seconds / form`**.

---

## 5. End-to-End System Architecture

### 5.1 Morphological Preprocessing & Smart Segmentation (`src/field_reader/segmenter.py`)
- **Comb-Box Grid Separation:** Detects vertical grid dividers and suppresses horizontal bounding box lines via connected component aspect-ratio filtering ($w > 60\%$ of cell with $h \le 3\text{px}$), preventing box borders from shrinking digits into punctuation specks.
- **Freeform Underline Suppression:** Extracts and subtracts continuous horizontal underline strokes using rectangular structuring elements.
- **Smart Box Merging:** Merges fragmented cursive strokes (e.g. the two vertical stems of `'U'` or the dot/tail of `'J'`) based on horizontal pitch overlap ($< 6\text{px}$ gap within typical character width), eliminating over-split index shifts.
- **Centroid Normalization:** Resizes character crops preserving aspect ratio, centering them into $32 \times 32$ float tensors with normalized $[0, 1]$ intensity.

### 5.2 Custom Character Classifier CNN (`src/field_reader/model.py`)
- **Architecture:** 4-layer convolutional feature extractor with BatchNorm, ReLU, Dropout (0.25), and dual dense classification heads outputting log-probabilities across 39 classes:
  - Digits: `0-9` (10 classes)
  - Uppercase Letters: `A-Z` (26 classes)
  - Punctuation & Separators: `/`, `-`, `.` (3 classes)
- **Stroke Morphology Augmentation:** Trained on 27,300 glyphs incorporating random morphological dilation (simulating ink bleed) and erosion (faint pencil) with adaptive Otsu binarization on an NVIDIA RTX 4060 GPU.

### 5.3 FSM Grammar & Lexical Dictionary Decoder (`src/field_reader/decoder.py`)
- **ISO Calendar Century Clamping:** Snaps cursive `'2'` vs `'9'` confusion in year positions (`90xx` $\rightarrow$ `20xx`, `99xx` $\rightarrow$ `19xx`), preventing impossible years like `9006`.
- **Calendar Range Enforcement:** Validates day ($01 \le DD \le 31$), month ($01 \le MM \le 12$), and repairs invalid month `00` to `10`.
- **Date Separator Consistency:** Forces identical separators across position 2 and position 5 (strictly both `/` or both `-`).
- **Postal PIN Directory Beam Search:** Scores 6-digit candidate probability lattices against registered postal directories using log-likelihood beam search with visual confusion penalties ($2 \leftrightarrow 8$, $2 \leftrightarrow 9$, $1 \leftrightarrow 7$).
- **Prefix Registry for Short Codes:** Matches 2-to-3 character letter prefixes against registered departmental codes (`[A-Z]{2,3}-\d{4}`), eliminating visual confusions like `'O'` for `'Q'` and `'D'` for `'P'`.

### 5.4 Semantic Lattice Verifier (`src/field_reader/semantic_verifier.py`)
- Validates field-level syntax consistency, computes calibrated confidence scores, and generates human-readable audit explanations for any automatic corrections applied.

---

## 6. Interactive Web Interface (`ocr_standalone_app.py`)
- **Local Address:** `http://127.0.0.1:7861` (Live Gradio Application)
- **Multi-Engine Mode Selector:**
  - `🚀 SOTA Tri-Engine Pipeline (Recommended)`
  - `⚡ Tier 1: CNN + Grammar/FSM Decoder`
  - `🔬 Baseline: Raw Character CNN`
- **Automated Gating Banners:**
  - 🟢 `✓ AUTOMATED DIGITIZATION APPROVED` when $C(F) \ge \theta$.
  - ⚠️ `FLAGGED FOR MANUAL VERIFICATION` when $C(F) < \theta$ or syntax violation detected.
- **Visual Glyph Strip:** Displays each individual $32 \times 32$ normalized character patch with its predicted character, confidence badge (Green $\ge 90\%$, Amber $75\text{--}89\%$, Red $< 75\%$), and top alternative candidate probabilities.
- **1-Click Operator Verification:** Editable transcription field with "✓ Accept & Confirm", "✏️ Submit Correction", and "🚩 Reject Field".
- **Persistent Audit Logging:** Logs Audit ID, timestamp, field type, raw prediction, verified text, and action taken.
- **Live Leaderboard Tab:** Visualizes comparative metrics and confidence trigger trade-offs directly in the UI.

---

## 7. Clean Repository File Manifest

```
C:\Users\SRIRAM\Documents\GitHub\OCR features for Hackathon\
├── data\
│   └── form_fields\
│       ├── metadata.csv          # Ground-truth labels & metadata for 150 test fields
│       └── field_*.png           # Cropped form field images (Dates, PINs, Codes)
├── models\
│   ├── field_cnn.pth             # Trained CNN model weights (99.95% val accuracy)
│   ├── field_cnn_history.json    # Loss & accuracy convergence curves
│   └── evaluation_report.json    # Official benchmark evaluation report
├── src\
│   ├── __init__.py               # Source package initializer
│   └── field_reader\
│       ├── __init__.py           # Field reader package initializer
│       ├── dataset.py            # Field generator & PyTorch CharacterDataset
│       ├── decoder.py            # FormFieldGrammarDecoder (Century & Lexicon FSM)
│       ├── model.py              # FieldCharacterCNN architecture & predict methods
│       ├── pipeline.py           # Unified multi-mode FormReaderPipeline
│       ├── segmenter.py          # Morphological segmenter & 32x32 glyph normalizer
│       └── semantic_verifier.py  # SemanticFieldVerifier (Lattice repair)
├── context.md                    # Single source of truth project documentation
├── evaluate_field_reader.py      # Comparative benchmark leaderboard runner
├── ocr_standalone_app.py         # Dedicated Gradio Operator Web App (Port 7861)
├── train_field_cnn.py            # Character CNN training script
├── requirements.txt              # Lean project dependencies
├── README.md                     # Clean project documentation & overview
└── .gitignore                    # Git ignore file
```


---

## 8. GitHub Repository & Deployment Details

- **GitHub Repository:** [https://github.com/Sriram-2090/OCR-features](https://github.com/Sriram-2090/OCR-features)
- **Repository Name:** OCR-features (Display Name: **OCR features**)
- **Target Branch:** main
- **Remote Origin URL:** https://github.com/Sriram-2090/OCR-features.git
- **Repository Type:** Standalone, dedicated repository containing exclusively Track B Form Field OCR deliverables.
- **Git Author:** Sriram-2090 (gsriram209@gmail.com)
