# -*- coding: utf-8 -*-
"""
ECG unimodal SQIs + 3-class SVM (RBF) (OOF-CV)

Purpose
-------
Train and evaluate a 3-class RBF-kernel SVM classifier using unimodal ECG SQIs
(previously extracted from raw ECG) and manual ECG quality labels ("ECG" column in the
manual annotation Excel).

The manual 6-point ECG scale is mapped into 3 classes:
    0           -> low_quality
    1, 2        -> uncertain
    3, 4, 5     -> high_quality

Evaluation protocol
-------------------
All reported metrics and per-sample predictions are computed from OUT-OF-FOLD (OOF)
predictions using StratifiedKFold cross-validation. Each sample is predicted exactly once
by a model that did not see it during training.

Artifacts generated (all OOF-based)
-----------------------------------
1) Confusion matrix (row-normalized %) saved as PNG.
2) Cross-validated macro metrics (mean ± std across folds) saved as CSV.
3) Cross-validated per-class metrics (mean ± std across folds) saved as CSV.
4) Per-sample OOF predictions saved as CSV with the structure:
   [ID, Auscultation_Point, ...ecg_sqis..., manual_label, predicted_label]

Inputs (relative to this script)
--------------------------------
- AQ_PATH: CSV with extracted ECG SQIs (expects columns 'ID', 'Auscultation_Point', and the SQI columns).
- MQ_PATH: Excel with manual labels (expects columns 'Trial', 'Spot', and 'ECG').

Notes
-----
- Merge rule (project convention):
    features.ID (as string) == manual.Trial (as string)
    features.Auscultation_Point == manual.Spot after normalization:
        remove '_' and convert to uppercase.

@author: Daniel Proaño-Guevara
"""

# %% Imports
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
    from sklearn.svm import SVC
except Exception as e:
    raise ImportError(
        "scikit-learn is required for this script. "
        f"Install it and rerun. Root error: {e}"
    )

warnings.filterwarnings("ignore", category=UserWarning)

# %% Global config (edit if your paths differ)
# This script is intended to live at:
#   Multimodal_SQA/tbme_code/3_unimodal_ecg/ecg_svm.py
# so the extracted features are in:
#   Multimodal_SQA/tbme_code/3_unimodal_ecg/exp_step3_ecg_features/step3_ecg_unimodal_features_extracted.csv
AQ_PATH = os.path.join(
    os.path.dirname(__file__),
    "exp_step3_ecg_features",
    "step3_ecg_unimodal_features_extracted.csv",
)

# Manual (annotated) labels file (as used in other project scripts)
MQ_PATH = r"..\..\..\ulsge_manual_sqa.xlsx"

# Output folder for all artifacts produced by this script
OUT_DIR = "exp_step3_ecg_unimodal_allSQI_svm_3class"
os.makedirs(OUT_DIR, exist_ok=True)

# Reproducibility and CV configuration
RANDOM_STATE = 42
N_SPLITS_CV = 5

# --- Manual label remapping to 3 classes (must match project convention) ---
RELABEL_MAP = {
    0: "low_quality",
    1: "uncertain",
    2: "uncertain",
    3: "high_quality",
    4: "high_quality",
    5: "high_quality",
}

# Fixed display / encoding order for classes
CLASS_ORDER = ["low_quality", "uncertain", "high_quality"]

# SQI feature columns (must match Step 3 extractor output)
ECG_SQI_COLS = [
    "bSQI",
    "pSQI",
    "sSQI",
    "kSQI",
    "fSQI",
    "basSQI",
]


# %% Utilities


def merge_quality_dataframes(ex1_quality: pd.DataFrame, m_quality: pd.DataFrame) -> pd.DataFrame:
    """
    Merge extracted ECG SQIs (ex1_quality) with manual labels (m_quality).

    Join keys
    ---------
    - ex1_quality.ID (string)         == m_quality.Trial (string)
    - ex1_quality.Auscultation_Point  == m_quality.Spot after normalization:
        * remove '_' characters
        * convert to uppercase

    Output
    ------
    Returns a clean dataframe with:
      - ID, Auscultation_Point
      - manual label: ECG (if present)
      - the SQI feature columns listed in ECG_SQI_COLS (if present)
    """
    ex1 = ex1_quality.copy()
    m = m_quality.copy()

    ex1["ID"] = ex1["ID"].astype(str)
    m["Trial"] = m["Trial"].astype(str)

    ex1["_k_point"] = (
        ex1["Auscultation_Point"]
        .astype(str)
        .str.replace("_", "", regex=False)
        .str.upper()
    )
    m["_k_spot"] = (
        m["Spot"]
        .astype(str)
        .str.replace("_", "", regex=False)
        .str.upper()
    )

    manual_keep = ["Trial", "_k_spot"]
    if "ECG" in m.columns:
        manual_keep.append("ECG")

    merged = pd.merge(
        ex1,
        m[manual_keep],
        left_on=["ID", "_k_point"],
        right_on=["Trial", "_k_spot"],
        how="inner",
    )

    sqi_cols = [c for c in ECG_SQI_COLS if c in merged.columns]
    base_cols = ["ID", "Auscultation_Point"] + (["ECG"] if "ECG" in merged.columns else [])
    return merged[base_cols + sqi_cols].copy()


def compute_specificity_from_cm(cm: np.ndarray) -> np.ndarray:
    """Per-class specificity from a multiclass confusion matrix."""
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    with np.errstate(divide="ignore", invalid="ignore"):
        spec = np.where((tn + fp) > 0, tn / (tn + fp), 0.0)
    return spec


def save_confusion_matrix_artifacts(y_true, y_pred, labels, display_labels, out_prefix: str) -> None:
    """Save ONLY the row-normalized confusion matrix as a PNG (OOF-based)."""
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_percent = np.where(row_sums > 0, cm_counts.astype(float) / row_sums * 100.0, 0.0)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=display_labels)
    disp.plot(ax=ax, values_format=".1f", cmap="Blues", colorbar=True)
    ax.set_title("Step 3 — SVM (ECG unimodal SQIs) — OOF Confusion Matrix (%)")
    plt.tight_layout()

    png_path = os.path.join(OUT_DIR, f"{out_prefix}_confusion_matrix_percent.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fold_metrics_from_predictions(y_true, y_pred, y_proba, class_labels) -> dict:
    """
    Compute fold-level metrics from predictions produced on the fold's held-out set.
    Requires y_proba for multiclass ROC-AUC (OvR).
    """
    y_bin = label_binarize(y_true, classes=class_labels)
    auc_ovr = roc_auc_score(y_bin, y_proba, multi_class="ovr")

    acc = accuracy_score(y_true, y_pred)
    macro_sens = recall_score(y_true, y_pred, average="macro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    spec_per_class = compute_specificity_from_cm(cm)
    macro_spec = float(np.mean(spec_per_class))

    prec_pc = precision_score(y_true, y_pred, average=None, labels=class_labels, zero_division=0)
    rec_pc = recall_score(y_true, y_pred, average=None, labels=class_labels, zero_division=0)
    f1_pc = f1_score(y_true, y_pred, average=None, labels=class_labels, zero_division=0)
    support_pc = cm.sum(axis=1).astype(int)

    return {
        "auc_ovr": float(auc_ovr),
        "accuracy": float(acc),
        "macro_sensitivity": float(macro_sens),
        "macro_specificity": float(macro_spec),
        "macro_f1": float(macro_f1),
        "per_class_precision": prec_pc.astype(float),
        "per_class_recall": rec_pc.astype(float),
        "per_class_specificity": spec_per_class.astype(float),
        "per_class_f1": f1_pc.astype(float),
        "per_class_support": support_pc,
    }


def mean_std(x: np.ndarray) -> tuple[float, float]:
    """Compute mean and sample standard deviation (ddof=1) for fold aggregates."""
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x)) if x.size else 0.0
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return mu, sd


# %% Load + merge
ex1_quality = pd.read_csv(AQ_PATH)
m_quality = pd.read_excel(MQ_PATH)

merged = merge_quality_dataframes(ex1_quality, m_quality)

if "ECG" not in merged.columns:
    raise ValueError("ECG label not found in merged dataframe. Ensure manual Excel contains 'ECG'.")

ecg_sqi_cols_present = [c for c in ECG_SQI_COLS if c in merged.columns]
if not ecg_sqi_cols_present:
    raise ValueError(
        "No expected ECG SQI columns found after merge. "
        f"Expected at least one of: {ECG_SQI_COLS}"
    )

# %% Label preprocessing
df = merged.copy()

df["ECG"] = pd.to_numeric(df["ECG"], errors="coerce")
df = df.dropna(subset=["ECG"]).copy()

df["manual_label"] = df["ECG"].astype(int).map(RELABEL_MAP)
df = df.dropna(subset=["manual_label"]).copy()

df = df.dropna(subset=ecg_sqi_cols_present).copy()

le = LabelEncoder()
le.fit(CLASS_ORDER)
df["y"] = le.transform(df["manual_label"].values)

X = df[ecg_sqi_cols_present].values
y = df["y"].values.astype(int)

df["ID"] = df["ID"].astype(str)
df["Auscultation_Point"] = df["Auscultation_Point"].astype(str)

class_labels = np.array([0, 1, 2], dtype=int)
class_names = CLASS_ORDER

# %% Model pipeline (RBF SVM)
# SVM with RBF kernel is sensitive to the scale of input features.
# StandardScaler standardizes each feature to zero-mean/unit-variance before the SVM.
#
# probability=True enables predict_proba(), which is required here for multiclass ROC-AUC.
# class_weight='balanced' reweights samples inversely to class frequency.
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )),
])

# %% Cross-validated training (OUT-OF-FOLD predictions)
skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

n = len(df)
oof_pred = np.full(shape=(n,), fill_value=-1, dtype=int)
oof_proba = np.full(shape=(n, len(class_labels)), fill_value=np.nan, dtype=float)

fold_macro = {
    "auc_ovr": [],
    "accuracy": [],
    "macro_sensitivity": [],
    "macro_specificity": [],
    "macro_f1": [],
}
fold_per_class = {
    "precision": [],
    "recall": [],
    "specificity": [],
    "f1": [],
    "support": [],
}

for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    svm_pipe.fit(Xtr, ytr)
    yhat = svm_pipe.predict(Xte)
    yproba = svm_pipe.predict_proba(Xte)

    oof_pred[te_idx] = yhat
    oof_proba[te_idx, :] = yproba

    fm = fold_metrics_from_predictions(
        y_true=yte,
        y_pred=yhat,
        y_proba=yproba,
        class_labels=class_labels,
    )

    for k in fold_macro:
        fold_macro[k].append(fm[k])

    fold_per_class["precision"].append(fm["per_class_precision"])
    fold_per_class["recall"].append(fm["per_class_recall"])
    fold_per_class["specificity"].append(fm["per_class_specificity"])
    fold_per_class["f1"].append(fm["per_class_f1"])
    fold_per_class["support"].append(fm["per_class_support"])

if np.any(oof_pred < 0) or np.any(~np.isfinite(oof_proba)):
    raise RuntimeError(
        "OOF predictions/probabilities are incomplete. "
        "Check that StratifiedKFold ran over all samples."
    )

# %% 1) Confusion matrix artifacts (OOF)
save_confusion_matrix_artifacts(
    y_true=y,
    y_pred=oof_pred,
    labels=class_labels,
    display_labels=class_names,
    out_prefix="step3_ecg_svm_cv",
)

# %% 2) Cross-validated macro metrics (mean ± std)
macro_rows = []
for metric_name, values in fold_macro.items():
    mu, sd = mean_std(np.array(values, dtype=float))
    macro_rows.append({"metric": metric_name, "mean": mu, "std": sd})

macro_df = pd.DataFrame(macro_rows)
macro_csv = os.path.join(OUT_DIR, "step3_ecg_svm_cv_macro_metrics.csv")
macro_df.to_csv(macro_csv, index=False)

# %% 3) Cross-validated per-class metrics (mean ± std)
pc_prec = np.vstack(fold_per_class["precision"])
pc_rec = np.vstack(fold_per_class["recall"])
pc_spec = np.vstack(fold_per_class["specificity"])
pc_f1 = np.vstack(fold_per_class["f1"])

pc_support_total = np.sum(np.vstack(fold_per_class["support"]), axis=0).astype(int)

per_class_rows = []
for ci, cname in enumerate(class_names):
    prec_mu, prec_sd = mean_std(pc_prec[:, ci])
    rec_mu, rec_sd = mean_std(pc_rec[:, ci])
    spec_mu, spec_sd = mean_std(pc_spec[:, ci])
    f1_mu, f1_sd = mean_std(pc_f1[:, ci])

    per_class_rows.append({
        "class": cname,
        "precision_mean": prec_mu, "precision_std": prec_sd,
        "recall_mean": rec_mu, "recall_std": rec_sd,
        "specificity_mean": spec_mu, "specificity_std": spec_sd,
        "f1_mean": f1_mu, "f1_std": f1_sd,
        "support_total": int(pc_support_total[ci]),
    })

per_class_df = pd.DataFrame(per_class_rows)
per_class_csv = os.path.join(OUT_DIR, "step3_ecg_svm_cv_per_class_metrics.csv")
per_class_df.to_csv(per_class_csv, index=False)

# %% 4) Per-sample OOF predictions CSV
pred_df = df[["ID", "Auscultation_Point"] + ecg_sqi_cols_present + ["manual_label"]].copy()
pred_df["predicted_label"] = np.array(class_names, dtype=object)[oof_pred]

pred_df = pred_df[["ID", "Auscultation_Point"] + ecg_sqi_cols_present + ["manual_label", "predicted_label"]]
pred_csv = os.path.join(OUT_DIR, "step3_ecg_svm_oof_predictions.csv")
pred_df.to_csv(pred_csv, index=False)

# %% Minimal console summary
print(f"[Step 3] Saved outputs to: {OUT_DIR}")
print(" - Confusion matrix (% PNG):    step3_ecg_svm_cv_confusion_matrix_percent.png")
print(" - CV macro metrics:            step3_ecg_svm_cv_macro_metrics.csv")
print(" - CV per-class metrics:        step3_ecg_svm_cv_per_class_metrics.csv")
print(" - OOF predictions (per-row):   step3_ecg_svm_oof_predictions.csv")
