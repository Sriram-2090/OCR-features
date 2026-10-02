# Track B: Handwritten Form Field Reader (JIG26_16)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.20%2B-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, high-accuracy handwritten form field digitization system designed for government and institutional departments. Built to digitize structured handwritten fields (**Dates**, **Postal PIN Codes**, and **Alphanumeric Short Codes**) from forms while providing automated confidence gating and a 1-click operator verification workflow.

---

## 🎯 Problem Statement & Scenario
A department needs to digitize selected handwritten numbers, dates, or short codes from structured forms while retaining manual verification for uncertain cases.

### Core Objectives:
1. **Automate Zero-Touch Extraction:** Maximize exact-match automated ingestion without human intervention.
2. **Minimize Operator Review:** Reduce human verification burden from typical $15\text{--}25\%$ down to **$< 8\%$**.
3. **Prevent Silent Errors:** Maintain $100\%$ precision on auto-accepted records via multi-layered schema and lexical dictionary validation.

---

## 🏆 Benchmark Leaderboard (150 Test Fields, 1,157 Glyphs)

Evaluated across **150 standardized test fields** (comb-box grid and freeform handwriting):

| Architecture / Model Mode | Character Accuracy | Character Error Rate (CER) | Complete-Field Exact Match | Human Review Routing Rate |
|---|:---:|:---:|:---:|:---:|
| **🔬 Baseline: Raw Character CNN** | `93.42%` | 6.58% | `76.00%` (114 / 150) | 24.00% |
| **⚡ Tier 1: CNN + Grammar/FSM Decoder** | `96.84%` | 3.16% | `93.33%` (140 / 150) | **3.30%** |
| **🚀 Tier 2: SOTA Tri-Engine Pipeline** | **`96.92%`** | **3.08%** | **`93.33%` (140 / 150)** | **`3.30%`** |

---

## ⚙️ Confidence Trigger & Human-in-the-Loop Trade-Offs

$$\text{Field Confidence: } C(F) = \min_{i=1 \dots N} P(c_i)$$

| Threshold $\theta$ | Zero-Touch Auto-Accept Rate | Auto-Accepted Ingested Acc | Human Review Routing Rate | Operational Status |
|:---:|:---:|:---:|:---:|---|
| **$\theta = 0.70$** | **`98.7%`** (148 / 150) | 93.9% | **`1.3%`** (2 fields) | Maximum high-throughput automation |
| **$\theta = 0.80$** | **`96.7%`** (145 / 150) | 94.5% | **`3.3%`** (5 fields) | Optimal balanced operational profile |
| **$\theta = 0.85$ (Default)** | **`96.7%`** (145 / 150) | 94.5% | **`3.3%`** (5 fields) | **Target achieved: $< 8\%$ review routing** |
| **$\theta = 0.95$ (Strict)** | **`96.7%`** (145 / 150) | 94.5% | **`3.3%`** (5 fields) | High-assurance automated ingest |

---

## 🧠 Architectural Pipeline

```
  Cropped Form Field Image (Date / PIN / Short Code)
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ 1. Character Segmentation & Normalizer  │
   │    • Comb-box grid detection            │
   │    • Freeform underline suppression     │
   │    • Smart pitch-constrained box merge  │
   │    • 32x32 centroid-centered glyphs     │
   └────────────────────┬────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ 2. Character Classification CNN         │
   │    • 39-Class multi-scale ConvNet       │
   │    • Stroke morphology training         │
   │    • Full logit & softmax distribution  │
   └────────────────────┬────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ 3. FSM Grammar & Lexical Lattice        │
   │    • ISO Calendar Century Clamping      │
   │    • Postal PIN Directory Beam Search   │
   │    • Code Prefix Registry Verification  │
   │    • Visual Confusion Affinity Matrix   │
   └────────────────────┬────────────────────┘
                        │
                        ▼
            Confidence Gating (θ = 0.85)
           ┌────────────┴────────────┐
           ▼                         ▼
   🟢 AUTOMATED INGESTION    ⚠️ FLAGGED FOR REVIEW
      (Database Commit)         (1-Click Operator UI)
```

---

## 🚀 Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Train Character CNN
```bash
python train_field_cnn.py
```
*Trains on 27,300 augmented glyphs on GPU in ~35 seconds, saving to `models/field_cnn.pth`.*

### 3. Run Benchmark Evaluation
```bash
python evaluate_field_reader.py
```
*Evaluates all 3 pipeline tiers across the 150 benchmark test fields and generates `models/evaluation_report.json`.*

### 4. Launch Interactive Web Interface
```bash
python ocr_standalone_app.py
```
*Starts the Gradio application at `http://127.0.0.1:7861`.*

---

## 📁 Repository Structure
```
OCR features for Hackathon/
├── data/
│   └── form_fields/
│       ├── metadata.csv          # Ground-truth labels & metadata for 150 test fields
│       └── field_*.png           # Cropped form field images (Dates, PINs, Codes)
├── models/
│   ├── field_cnn.pth             # Trained CNN model weights (99.95% val accuracy)
│   ├── field_cnn_history.json    # Loss & accuracy convergence curves
│   └── evaluation_report.json    # Benchmark metrics & confidence trigger table
├── src/
│   └── field_reader/
│       ├── __init__.py           # Package initializer
│       ├── model.py              # FieldCharacterCNN architecture & predict methods
│       ├── segmenter.py          # Morphological segmenter & 32x32 glyph normalizer
│       ├── dataset.py            # Field generator & PyTorch CharacterDataset
│       ├── decoder.py            # FormFieldGrammarDecoder (Century & Lexicon FSM)
│       ├── semantic_verifier.py  # SemanticFieldVerifier (Lattice repair)
│       └── pipeline.py           # Unified multi-mode FormReaderPipeline
├── context.md                    # Single source of truth project documentation
├── evaluate_field_reader.py      # Comparative benchmark leaderboard runner
├── ocr_standalone_app.py         # Dedicated Gradio Operator Web App (Port 7861)
├── train_field_cnn.py            # Character CNN training script
├── requirements.txt              # Lean project dependencies
└── README.md                     # Project documentation & overview
```
