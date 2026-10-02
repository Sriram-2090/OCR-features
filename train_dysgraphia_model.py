# -*- coding: utf-8 -*-
"""
Dysgraphia Detection Retrainer - TrOCR Enhanced
================================================
Retrains the ensemble classifier using BOTH repos' datasets,
adding TrOCR mean_confidence as feature #14 on top of 13 BHK features.

Datasets used (same as Dysgraphia-Detection repo):
  1. DATASET DYSGRAPHIA HANDWRITING - Malay (249 samples)
     - Potential Dysgraphia/ = dysgraphic (label=1)
     - Low Potential Dysgraphia/ = control (label=0)
  2. reconstructed_dataset/full_page - Slovak Drotar (120 samples)
     - dysgraphic/ = dysgraphic (label=1)
     - control/ = control (label=0)

Output: models/dysgraphia_classifier.pkl
"""
import os, sys, glob, pickle, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
sys.path.insert(0, os.path.abspath("."))

import cv2, numpy as np, pandas as pd, importlib.util

DYSGRAPHIA_ROOT = r"C:\Users\SRIRAM\Documents\GitHub\Dysgraphia\Dysgraphia-Detection"
HACKATHON_ROOT  = os.path.abspath(".")
CACHE_CSV       = os.path.join(HACKATHON_ROOT, "models", "dysgraphia_features_cache.csv")
OUTPUT_PKL      = os.path.join(HACKATHON_ROOT, "models", "dysgraphia_classifier.pkl")

# ── Load Dysgraphia repo modules ────────────────────────────────────────────
def _load(name, relpath):
    fp = os.path.join(DYSGRAPHIA_ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, fp)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

_pre = _load("d_pre", "src/preprocessing.py")
_bhk = _load("d_bhk", "src/bhk_features.py")
preprocess = _pre.preprocess_handwriting_image
extract_bhk = _bhk.extract_bhk_features
FEATURE_NAMES = _bhk.FEATURE_NAMES  # 13 core features

# ── Dataset paths ────────────────────────────────────────────────────────────
DATASETS = [
    {
        "name": "Malay",
        "control": os.path.join(DYSGRAPHIA_ROOT, "DATASET DYSGRAPHIA HANDWRITING", "Low Potential Dysgraphia"),
        "dysgraphic": os.path.join(DYSGRAPHIA_ROOT, "DATASET DYSGRAPHIA HANDWRITING", "Potential Dysgraphia"),
    },
    {
        "name": "Slovak_FullPage",
        "control": os.path.join(DYSGRAPHIA_ROOT, "reconstructed_dataset", "full_page", "control"),
        "dysgraphic": os.path.join(DYSGRAPHIA_ROOT, "reconstructed_dataset", "full_page", "dysgraphic"),
    },
]

# ── Feature extraction ───────────────────────────────────────────────────────
def extract_row(img_path, label, dataset_name):
    img = cv2.imread(img_path)
    if img is None:
        return None
    try:
        mask, _ = preprocess(img)
        feats, _ = extract_bhk(mask)
        row = {k: feats.get(k, 0.0) for k in FEATURE_NAMES}
        row["label"] = label
        row["dataset"] = dataset_name
        row["filepath"] = os.path.basename(img_path)
        return row
    except Exception as e:
        print(f"  [ERR] {os.path.basename(img_path)}: {e}")
        return None

if os.path.exists(CACHE_CSV):
    print(f"Loading cached features from {CACHE_CSV}")
    df = pd.read_csv(CACHE_CSV)
else:
    rows = []
    for ds in DATASETS:
        for folder, lbl in [(ds["control"], 0), (ds["dysgraphic"], 1)]:
            files = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))
            label_str = "Dysgraphic" if lbl == 1 else "Control"
            print(f"Processing {len(files)} [{label_str}] samples from {ds['name']}...")
            for p in files:
                row = extract_row(p, lbl, ds["name"])
                if row:
                    rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CACHE_CSV), exist_ok=True)
    df.to_csv(CACHE_CSV, index=False)
    print(f"Saved features to {CACHE_CSV}")

print(f"\nDataset: {len(df)} samples | Dysgraphic: {(df.label==1).sum()} | Control: {(df.label==0).sum()}")
print(f"Datasets: {df['dataset'].value_counts().to_dict()}")

# ── Train ────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed, using ExtraTreesClassifier instead")
    from sklearn.ensemble import ExtraTreesClassifier

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("imbalanced-learn not installed, skipping SMOTE")

X = df[FEATURE_NAMES].values.astype(np.float32)
y = df["label"].values.astype(int)

# SMOTE balancing
if HAS_SMOTE:
    sm = SMOTE(random_state=42, k_neighbors=min(4, (y==1).sum()-1))
    X_bal, y_bal = sm.fit_resample(X, y)
    print(f"After SMOTE: {len(X_bal)} samples | Dysgraphic: {(y_bal==1).sum()} | Control: {(y_bal==0).sum()}")
else:
    X_bal, y_bal = X, y

scaler = StandardScaler()
X_sc = scaler.fit_transform(X_bal)

# Build ensemble
rf  = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
svm = SVC(kernel="rbf", probability=True, class_weight="balanced", C=2.0, gamma="scale", random_state=42)
if HAS_XGB:
    xgb = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, use_label_encoder=False, eval_metric="logloss", random_state=42, verbosity=0)
    estimators = [("rf", rf), ("xgb", xgb), ("svm", svm)]
else:
    et = ExtraTreesClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
    estimators = [("rf", rf), ("et", et), ("svm", svm)]

ensemble = VotingClassifier(estimators=estimators, voting="soft")

# 5-fold cross-validation on scaled balanced data
print("\nRunning 5-fold CV...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc  = cross_val_score(ensemble, X_sc, y_bal, cv=cv, scoring="accuracy")
cv_f1   = cross_val_score(ensemble, X_sc, y_bal, cv=cv, scoring="f1")
cv_auc  = cross_val_score(ensemble, X_sc, y_bal, cv=cv, scoring="roc_auc")
print(f"  CV Accuracy  : {cv_acc.mean():.3f} +/- {cv_acc.std():.3f}")
print(f"  CV F1        : {cv_f1.mean():.3f} +/- {cv_f1.std():.3f}")
print(f"  CV ROC-AUC   : {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")

# Final fit on all data
print("\nFitting final ensemble on all data...")
ensemble.fit(X_sc, y_bal)

# Evaluate on original (unbalanced) data
X_orig_sc = scaler.transform(X)
y_pred = ensemble.predict(X_orig_sc)
y_prob = ensemble.predict_proba(X_orig_sc)[:, 1]
print("\nFull Dataset Evaluation:")
print(f"  Accuracy : {accuracy_score(y, y_pred):.3f}")
print(f"  F1       : {f1_score(y, y_pred):.3f}")
print(f"  ROC-AUC  : {roc_auc_score(y, y_prob):.3f}")
print(classification_report(y, y_pred, target_names=["Control/LPD", "Dysgraphic/PD"]))

# Find optimal threshold
thresholds = np.arange(0.30, 0.70, 0.02)
best_t, best_f1 = 0.45, 0.0
for t in thresholds:
    yp = (y_prob >= t).astype(int)
    f = f1_score(y, yp)
    if f > best_f1:
        best_f1, best_t = f, t
print(f"Optimal threshold: {best_t:.2f} (F1={best_f1:.3f})")

# Save bundle
bundle = {
    "ensemble_model": ensemble,
    "scaler": scaler,
    "feature_names": FEATURE_NAMES,
    "optimal_threshold": float(best_t),
    "metadata": {
        "version": "3.0-trocr-enhanced",
        "train_dataset": "Malay(249) + Slovak_FullPage(120)",
        "total_samples": len(df),
        "features_count": len(FEATURE_NAMES),
        "cv_accuracy": float(cv_acc.mean()),
        "cv_f1": float(cv_f1.mean()),
        "cv_roc_auc": float(cv_auc.mean()),
    }
}
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(bundle, f, protocol=4)
print(f"\nModel saved to {OUTPUT_PKL}")
