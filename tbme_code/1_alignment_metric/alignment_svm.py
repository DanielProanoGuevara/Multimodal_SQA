# -*- coding: utf-8 -*-
"""
Alignment metric (multimodal) + 3-class RBF SVM

Purpose
-------
Train and evaluate a 3-class Support Vector Machine (SVM) classifier with an RBF kernel
using only multimodal alignment-derived features (all columns starting with
'alignment_metric') and manual quality labels ('mSQA_min').

The manual 6-point scale is mapped into 3 classes:
    0           -> low_quality
    1, 2        -> uncertain
    3, 4, 5     -> high_quality

Evaluation protocol
-------------------
All reported metrics and the per-sample predictions are computed from
OUT-OF-FOLD (OOF) predictions using StratifiedKFold cross-validation.
This ensures each sample is predicted by a model that did not see it during
training (i.e., avoids optimistic bias).

Artifacts generated (all OOF-based)
-----------------------------------
1) Confusion matrix (row-normalized %) saved as PNG.
2) Cross-validated macro metrics (mean ± std across folds) saved as CSV.
3) Cross-validated per-class metrics (mean ± std across folds) saved as CSV.
4) Per-sample OOF predictions saved as CSV with the structure:
   [ID, Auscultation_Point, ...alignment_metrics..., manual_label, predicted_label]

Inputs (relative to this script)
--------------------------------
- AQ_PATH: Pickle with automatic metrics (expects columns 'ID', 'Auscultation_Point', and 'alignment_metric*').
- MQ_PATH: Excel with manual labels (expects columns 'Trial', 'Spot', and 'mSQA_min').

Notes
-----
- The merge is performed by:
    ex1_quality.ID (as string) == m_quality.Trial (as string)
    ex1_quality.Auscultation_Point == m_quality.Spot after normalization:
        remove '_' and convert to uppercase.
- SVM with RBF kernel is sensitive to feature scaling; therefore StandardScaler is applied
  before SVC.

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

# Ignore generic sklearn user warnings (does not silence FutureWarnings).
warnings.filterwarnings("ignore", category=UserWarning)

# %% Global config (edit if your paths differ)
# Automatic (computed) metrics file
AQ_PATH = r"..\..\..\ulsge_quality_metrics.pkl"
# Manual (annotated) labels file
MQ_PATH = r"..\..\..\ulsge_manual_sqa.xlsx"

# Output folder for all artifacts produced by this script
OUT_DIR = "exp_step1_svm_rbf_3class"
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


# %% Utilities


def merge_quality_dataframes(ex1_quality: pd.DataFrame, m_quality: pd.DataFrame) -> pd.DataFrame:
    """
    Merge automatic metrics (ex1_quality) with manual labels (m_quality).

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
      - manual labels (if present): mSQA_min, ECG, PCG
      - all columns starting with 'alignment_metric'
    """
    ex1 = ex1_quality.copy()
    m = m_quality.copy()

    # Ensure compatible join types
    ex1["ID"] = ex1["ID"].astype(str)
    m["Trial"] = m["Trial"].astype(str)

    # Normalized join keys (kept only during merge)
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

    # Manual columns to keep (only if they exist in the file)
    manual_keep = ["Trial", "_k_spot"]
    for c in ["mSQA_min", "ECG", "PCG"]:
        if c in m.columns:
            manual_keep.append(c)

    merged = pd.merge(
        ex1,
        m[manual_keep],
        left_on=["ID", "_k_point"],
        right_on=["Trial", "_k_spot"],
        how="inner",
    )

    # Keep only the alignment features (all columns starting with this prefix)
    alignment_cols = [c for c in ex1.columns if str(c).startswith("alignment_metric")]

    # Keep only known manual columns if present in the merged dataframe
    base_cols = ["ID", "Auscultation_Point"] + [
        c for c in ["mSQA_min", "ECG", "PCG"] if c in merged.columns
    ]

    return merged[base_cols + alignment_cols].copy()


def compute_specificity_from_cm(cm: np.ndarray) -> np.ndarray:
    """
    Compute per-class specificity from a multiclass confusion matrix.

    For a given class k:
        Specificity_k = TN_k / (TN_k + FP_k)

    Returns
    -------
    spec : ndarray of shape (n_classes,)
        Specificity for each class index.
    """
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    with np.errstate(divide="ignore", invalid="ignore"):
        spec = np.where((tn + fp) > 0, tn / (tn + fp), 0.0)

    return spec


def save_confusion_matrix_artifacts(y_true, y_pred, labels, display_labels, out_prefix: str) -> None:
    """
    Save ONLY the row-normalized confusion matrix as a PNG.

    The confusion matrix is computed on the provided vectors (here: OOF predictions),
    then normalized by row to show per-true-class percentages.
    """
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)

    # Row-normalize (i.e., normalize by true class count)
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_percent = np.where(
            row_sums > 0,
            cm_counts.astype(float) / row_sums * 100.0,
            0.0,
        )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=display_labels)
    disp.plot(ax=ax, values_format=".1f", cmap="Blues", colorbar=True)
    ax.set_title("Step 1 — RBF SVM (alignment metrics) — OOF Confusion Matrix (%)")
    plt.tight_layout()

    png_path = os.path.join(OUT_DIR, f"{out_prefix}_confusion_matrix_percent.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fold_metrics_from_predictions(y_true, y_pred, y_proba, class_labels) -> dict:
    """
    Compute fold-level metrics from predictions produced on the fold's held-out set.

    Macro metrics
    -------------
    - AUC OvR (multiclass one-vs-rest) computed from predicted probabilities
    - Accuracy
    - Macro Sensitivity (macro recall)
    - Macro Specificity (mean of per-class specificities)
    - Macro F1

    Per-class metrics
    -----------------
    - Precision, Recall, Specificity, F1, Support (counts)

    Returns
    -------
    dict with scalars for macro metrics and arrays for per-class metrics.
    """
    # AUC OvR: binarize y_true to (n_samples, n_classes) and pass predicted probabilities
    y_bin = label_binarize(y_true, classes=class_labels)
    auc_ovr = roc_auc_score(y_bin, y_proba, multi_class="ovr")

    acc = accuracy_score(y_true, y_pred)
    macro_sens = recall_score(y_true, y_pred, average="macro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    spec_per_class = compute_specificity_from_cm(cm)
    macro_spec = float(np.mean(spec_per_class))

    # Per-class PRF (specificity derived from confusion matrix)
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
    """
    Compute mean and sample standard deviation (ddof=1) for fold aggregates.

    If fewer than 2 folds are present, the std is returned as 0.0.
    """
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x)) if x.size else 0.0
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return mu, sd


# %% Step 1 — Load + merge
# Load automatic (computed) features and manual labels
ex1_quality = pd.read_pickle(AQ_PATH)
m_quality = pd.read_excel(MQ_PATH)

# Merge into a single table containing manual labels and alignment_metric* features
merged = merge_quality_dataframes(ex1_quality, m_quality)

# Manual labels are required for supervised training
if "mSQA_min" not in merged.columns:
    raise ValueError(
        "mSQA_min not found in merged dataframe. Ensure manual Excel contains 'mSQA_min'."
    )

# Select all alignment features by prefix
alignment_metrics = [c for c in merged.columns if str(c).startswith("alignment_metric")]
if not alignment_metrics:
    raise ValueError("No alignment_metric* columns found after merge.")

# %% Label preprocessing
# Convert manual labels to numeric (invalid entries become NaN) then drop missing labels
df = merged.copy()
df["mSQA_min"] = pd.to_numeric(df["mSQA_min"], errors="coerce")
df = df.dropna(subset=["mSQA_min"]).copy()

# Map the 6-point label into 3 classes, then drop unmapped / invalid entries
df["manual_label"] = df["mSQA_min"].astype(int).map(RELABEL_MAP)
df = df.dropna(subset=["manual_label"]).copy()

# Remove samples with missing feature values
df = df.dropna(subset=alignment_metrics).copy()

# Encode labels into integers using a fixed class order to ensure stable mapping across runs
le = LabelEncoder()
le.fit(CLASS_ORDER)
df["y"] = le.transform(df["manual_label"].values)

# Features and targets as numpy arrays for sklearn
X = df[alignment_metrics].values
y = df["y"].values.astype(int)

# IDs for the per-sample export (keep as strings for stable CSV formatting)
df["ID"] = df["ID"].astype(str)
df["Auscultation_Point"] = df["Auscultation_Point"].astype(str)

# Fixed class indices and names (must match LabelEncoder fit order)
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
    ))
])

# %% Cross-validated training (OUT-OF-FOLD predictions)
# StratifiedKFold preserves label proportions in each fold.
skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

# Allocate OUT-OF-FOLD prediction containers
n = len(df)
oof_pred = np.full(shape=(n,), fill_value=-1, dtype=int)
oof_proba = np.full(shape=(n, len(class_labels)), fill_value=np.nan, dtype=float)

# Fold-level aggregation (macro and per-class)
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
    "support": [],  # count per class in the fold test split
}

# Iterate folds and store predictions for the held-out portion each time
for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    # Fit model on the training split
    svm_pipe.fit(Xtr, ytr)

    # Predict class and class probabilities on the held-out split
    yhat = svm_pipe.predict(Xte)
    yproba = svm_pipe.predict_proba(Xte)

    # Store OUT-OF-FOLD predictions (each sample predicted exactly once)
    oof_pred[te_idx] = yhat
    oof_proba[te_idx, :] = yproba

    # Compute fold metrics
    fm = fold_metrics_from_predictions(
        y_true=yte,
        y_pred=yhat,
        y_proba=yproba,
        class_labels=class_labels,
    )

    # Store macro metrics for aggregation
    for k in fold_macro:
        fold_macro[k].append(fm[k])

    # Store per-class arrays for aggregation
    fold_per_class["precision"].append(fm["per_class_precision"])
    fold_per_class["recall"].append(fm["per_class_recall"])
    fold_per_class["specificity"].append(fm["per_class_specificity"])
    fold_per_class["f1"].append(fm["per_class_f1"])
    fold_per_class["support"].append(fm["per_class_support"])

# Sanity check: ensure every sample received exactly one OOF prediction + probability vector
if np.any(oof_pred < 0) or np.any(~np.isfinite(oof_proba)):
    raise RuntimeError(
        "OOF predictions/probabilities are incomplete. "
        "Check that StratifiedKFold ran over all samples."
    )

# %% 1) Confusion matrix artifacts (OOF)
# Uses OOF predictions so the confusion matrix reflects cross-validated performance.
save_confusion_matrix_artifacts(
    y_true=y,
    y_pred=oof_pred,
    labels=class_labels,
    display_labels=class_names,
    out_prefix="step1_alignment_svm_rbf_cv",
)

# %% 2) Cross-validated macro metrics (mean ± std across folds)
macro_rows = []
for metric_name, values in fold_macro.items():
    mu, sd = mean_std(np.array(values, dtype=float))
    macro_rows.append({"metric": metric_name, "mean": mu, "std": sd})

macro_df = pd.DataFrame(macro_rows)
macro_csv = os.path.join(OUT_DIR, "step1_alignment_svm_rbf_cv_macro_metrics.csv")
macro_df.to_csv(macro_csv, index=False)

# %% 3) Cross-validated per-class metrics (mean ± std across folds)
# Stack folds -> (n_folds, n_classes)
pc_prec = np.vstack(fold_per_class["precision"])
pc_rec = np.vstack(fold_per_class["recall"])
pc_spec = np.vstack(fold_per_class["specificity"])
pc_f1 = np.vstack(fold_per_class["f1"])

# Support totals are meaningful here because each sample appears in exactly one fold test set.
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
per_class_csv = os.path.join(OUT_DIR, "step1_alignment_svm_rbf_cv_per_class_metrics.csv")
per_class_df.to_csv(per_class_csv, index=False)

# %% 4) Per-sample OUT-OF-FOLD predictions CSV
# Required structure:
# [ID, Auscultation_Point, ...alignment_metrics..., manual_label, predicted_label]
pred_df = df[["ID", "Auscultation_Point"] + alignment_metrics + ["manual_label"]].copy()

# Convert predicted integer labels back to string labels using the fixed class order
pred_df["predicted_label"] = np.array(class_names, dtype=object)[oof_pred]

# Ensure the exact column order requested
pred_df = pred_df[["ID", "Auscultation_Point"] + alignment_metrics + ["manual_label", "predicted_label"]]

pred_csv = os.path.join(OUT_DIR, "step1_alignment_svm_rbf_oof_predictions.csv")
pred_df.to_csv(pred_csv, index=False)

# %% Minimal console summary
print(f"[Step 1] Saved outputs to: {OUT_DIR}")
print(" - Confusion matrix (% PNG):    step1_alignment_svm_rbf_cv_confusion_matrix_percent.png")
print(" - CV macro metrics:            step1_alignment_svm_rbf_cv_macro_metrics.csv")
print(" - CV per-class metrics:        step1_alignment_svm_rbf_cv_per_class_metrics.csv")
print(" - OOF predictions (per-row):   step1_alignment_svm_rbf_oof_predictions.csv")
