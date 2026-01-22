# -*- coding: utf-8 -*-
"""
Decision fusion (min rule) using SVM unimodal outputs + 3-class metrics (OOF-based)

File
----
Multimodal_SQA/tbme_code/4_decision_fusion/min_svm.py

Purpose
-------
Fuse unimodal ECG and PCG OUT-OF-FOLD (OOF) predictions produced by the *SVM* pipelines
(via the "minimum" rule), then evaluate against the multimodal manual label (mSQA_min).

Fusion rule
-----------
Predictions are assumed to use the project label encoding order:

    low_quality   -> 0
    uncertain     -> 1
    high_quality  -> 2

Decision fusion (min):
    y_fused = min(y_ecg_pred, y_pcg_pred)

Ground truth
------------
mSQA_min is loaded from ulsge_manual_sqa.xlsx and mapped into the project 3-class convention:
    0           -> low_quality
    1, 2        -> uncertain
    3, 4, 5     -> high_quality

Artifacts generated
-------------------
1) Confusion matrix (row-normalized %) saved as PNG.
2) Cross-validated macro metrics (mean ± std across folds) saved as CSV.
3) Cross-validated per-class metrics (mean ± std across folds) saved as CSV.
4) Per-sample fused predictions saved as CSV with the structure:
   [ID, Auscultation_Point, ecg_predicted_label, pcg_predicted_label, mSQA_min, manual_label, predicted_label]

IMPORTANT (per-sample CSV)
--------------------------
manual_label is the same as mSQA_min (raw 6-point scale duplicated for convenience).

Inputs (relative to this script)
--------------------------------
1) PCG SVM OOF predictions CSV (from Step 2 PCG SVM):
   ../2_unimodal_pcg/exp_step2_pcg_unimodal_allSQI_svm_3class/step2_pcg_svm_oof_predictions.csv

2) ECG SVM OOF predictions CSV (from Step 3 ECG SVM):
   ../3_unimodal_ecg/exp_step3_ecg_unimodal_allSQI_svm_3class/step3_ecg_svm_oof_predictions.csv

3) Manual labels Excel:
   ../../../ulsge_manual_sqa.xlsx

Merge rule (project convention)
-------------------------------
- ID == Trial (string)
- Auscultation_Point == Spot after normalization:
    remove '_' and convert to uppercase

Notes on AUC
------------
AUC requires probabilities. If your OOF CSVs include probability columns, this script can
be extended. By default it uses hard labels only, so AUC is exported as NaN.

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
    )
    from sklearn.model_selection import StratifiedKFold
except Exception as e:
    raise ImportError(
        "scikit-learn is required for this script. "
        f"Install it and rerun. Root error: {e}"
    )

warnings.filterwarnings("ignore", category=UserWarning)

# %% Global config
THIS_DIR = os.path.dirname(__file__)

# --- SVM OOF outputs (edit here if your filenames differ) ---
PCG_OOF_PATH = os.path.join(
    THIS_DIR,
    "..",
    "2_unimodal_pcg",
    "exp_step2_pcg_unimodal_allSQI_svm_rbf_3class",
    "step2_pcg_svm_rbf_oof_predictions.csv",
)

ECG_OOF_PATH = os.path.join(
    THIS_DIR,
    "..",
    "3_unimodal_ecg",
    "exp_step3_ecg_unimodal_allSQI_svm_3class",
    "step3_ecg_svm_oof_predictions.csv",
)

# Manual labels Excel
MQ_PATH = os.path.join(THIS_DIR, "..", "..", "..", "ulsge_manual_sqa.xlsx")

# Output folder for all artifacts produced by this script
OUT_DIR = "exp_step4_decision_fusion_min_svm_3class"
os.makedirs(OUT_DIR, exist_ok=True)

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
CLASS_TO_INT = {c: i for i, c in enumerate(CLASS_ORDER)}
INT_TO_CLASS = {i: c for i, c in enumerate(CLASS_ORDER)}

# Expected columns in OOF prediction files
REQ_OOF_COLS = ["ID", "Auscultation_Point", "predicted_label"]


# %% Utilities


def _norm_point(series: pd.Series) -> pd.Series:
    """Normalize auscultation points for robust joins: remove '_' and uppercase."""
    return series.astype(str).str.replace("_", "", regex=False).str.upper()


def compute_specificity_from_cm(cm: np.ndarray) -> np.ndarray:
    """Per-class specificity: TN_k / (TN_k + FP_k)."""
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    with np.errstate(divide="ignore", invalid="ignore"):
        spec = np.where((tn + fp) > 0, tn / (tn + fp), 0.0)
    return spec


def save_confusion_matrix_artifacts(y_true, y_pred, labels, display_labels, out_prefix: str) -> None:
    """Save ONLY the row-normalized confusion matrix as a PNG (computed on all samples)."""
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)

    row_sums = cm_counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_percent = np.where(row_sums > 0, cm_counts.astype(
            float) / row_sums * 100.0, 0.0)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_percent, display_labels=display_labels)
    disp.plot(ax=ax, values_format=".1f", cmap="Blues", colorbar=True)
    ax.set_title(
        "Step 4 — Decision fusion (min) — SVM OOF Confusion Matrix (%)")
    plt.tight_layout()

    png_path = os.path.join(
        OUT_DIR, f"{out_prefix}_confusion_matrix_percent.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fold_metrics_from_hard_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_labels: np.ndarray) -> dict:
    """Fold metrics from hard predictions only; AUC not available -> NaN."""
    acc = accuracy_score(y_true, y_pred)
    macro_sens = recall_score(y_true, y_pred, average="macro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    spec_per_class = compute_specificity_from_cm(cm)
    macro_spec = float(np.mean(spec_per_class))

    prec_pc = precision_score(
        y_true, y_pred, average=None, labels=class_labels, zero_division=0)
    rec_pc = recall_score(y_true, y_pred, average=None,
                          labels=class_labels, zero_division=0)
    f1_pc = f1_score(y_true, y_pred, average=None,
                     labels=class_labels, zero_division=0)
    support_pc = cm.sum(axis=1).astype(int)

    return {
        "auc_ovr": float(np.nan),
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


def _validate_oof_df(df: pd.DataFrame, name: str) -> None:
    """Fail-fast validation of the expected OOF input schema."""
    missing = [c for c in REQ_OOF_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} missing required columns {missing}. Found: {list(df.columns)}")


def merge_unimodal_oof(pcg_oof: pd.DataFrame, ecg_oof: pd.DataFrame) -> pd.DataFrame:
    """
    Merge PCG and ECG OOF predictions by:
      - ID + normalized Auscultation_Point
    Keeps:
      ID, Auscultation_Point, pcg_predicted_label, ecg_predicted_label
    """
    pcg = pcg_oof.copy()
    ecg = ecg_oof.copy()

    pcg["ID"] = pcg["ID"].astype(str)
    ecg["ID"] = ecg["ID"].astype(str)

    pcg["_k_point"] = _norm_point(pcg["Auscultation_Point"])
    ecg["_k_point"] = _norm_point(ecg["Auscultation_Point"])

    pcg = pcg.rename(columns={"predicted_label": "pcg_predicted_label"})
    ecg = ecg.rename(columns={"predicted_label": "ecg_predicted_label"})

    merged = pd.merge(
        pcg[["ID", "Auscultation_Point", "_k_point", "pcg_predicted_label"]].copy(),
        ecg[["ID", "Auscultation_Point", "_k_point", "ecg_predicted_label"]].copy(),
        on=["ID", "_k_point"],
        how="inner",
        suffixes=("_pcg", "_ecg"),
    )

    # Keep PCG's Auscultation_Point string (deterministic choice)
    if "Auscultation_Point_pcg" in merged.columns:
        merged = merged.rename(
            columns={"Auscultation_Point_pcg": "Auscultation_Point"})
        merged = merged.drop(columns=["Auscultation_Point_ecg"])

    merged = merged.drop(columns=["_k_point"])
    return merged[["ID", "Auscultation_Point", "ecg_predicted_label", "pcg_predicted_label"]].copy()


def merge_with_manual_msqa(fused_df: pd.DataFrame, manual_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fused table with manual labels (mSQA_min), using:
      fused.ID == manual.Trial (as str)
      fused.Auscultation_Point == manual.Spot normalized

    Returns:
      [ID, Auscultation_Point, ecg_predicted_label, pcg_predicted_label, mSQA_min]
    """
    f = fused_df.copy()
    m = manual_df.copy()

    for c in ["Trial", "Spot", "mSQA_min"]:
        if c not in m.columns:
            raise ValueError(
                f"Manual Excel missing required column '{c}'. Found columns: {list(m.columns)}"
            )

    f["ID"] = f["ID"].astype(str)
    m["Trial"] = m["Trial"].astype(str)

    f["_k_point"] = _norm_point(f["Auscultation_Point"])
    m["_k_spot"] = _norm_point(m["Spot"])

    out = pd.merge(
        f,
        m[["Trial", "_k_spot", "mSQA_min"]].copy(),
        left_on=["ID", "_k_point"],
        right_on=["Trial", "_k_spot"],
        how="inner",
    )

    out = out.drop(columns=["_k_point", "_k_spot", "Trial"])
    return out[
        ["ID", "Auscultation_Point", "ecg_predicted_label",
            "pcg_predicted_label", "mSQA_min"]
    ].copy()


# %% Load inputs
pcg_oof = pd.read_csv(PCG_OOF_PATH)
ecg_oof = pd.read_csv(ECG_OOF_PATH)
m_quality = pd.read_excel(MQ_PATH)

_validate_oof_df(pcg_oof, "PCG SVM OOF predictions")
_validate_oof_df(ecg_oof, "ECG SVM OOF predictions")

# %% Merge + align unimodal OOF predictions
df = merge_unimodal_oof(pcg_oof, ecg_oof)
print(f"[Step 4 - SVM] Joined unimodal OOF rows (ECG+PCG): {len(df)}")

# %% Merge with manual mSQA_min labels
df = merge_with_manual_msqa(df, m_quality)
print(f"[Step 4 - SVM] Joined with manual mSQA_min rows: {len(df)}")

# %% Build 3-class ground truth from mSQA_min (used for metrics)
df["mSQA_min"] = pd.to_numeric(df["mSQA_min"], errors="coerce")
df = df.dropna(subset=["mSQA_min"]).copy()

df["gt_3class_str"] = df["mSQA_min"].astype(int).map(RELABEL_MAP)
df = df.dropna(subset=["gt_3class_str"]).copy()

df["y_true"] = df["gt_3class_str"].map(CLASS_TO_INT)
df = df.dropna(subset=["y_true"]).copy()
df["y_true"] = df["y_true"].astype(int)

# %% Convert unimodal predictions to ints and fuse via min
df["ecg_pred_int"] = df["ecg_predicted_label"].astype(str).map(CLASS_TO_INT)
df["pcg_pred_int"] = df["pcg_predicted_label"].astype(str).map(CLASS_TO_INT)

df = df.dropna(subset=["ecg_pred_int", "pcg_pred_int"]).copy()
df["ecg_pred_int"] = df["ecg_pred_int"].astype(int)
df["pcg_pred_int"] = df["pcg_pred_int"].astype(int)

df["y_pred"] = np.minimum(df["ecg_pred_int"].values,
                          df["pcg_pred_int"].values).astype(int)

class_labels = np.array([0, 1, 2], dtype=int)
class_names = CLASS_ORDER

# %% 1) Confusion matrix (computed on all samples)
save_confusion_matrix_artifacts(
    y_true=df["y_true"].values,
    y_pred=df["y_pred"].values,
    labels=class_labels,
    display_labels=class_names,
    out_prefix="step4_min_fusion_svm",
)

# %% 2) "CV-style" macro metrics (mean ± std across folds)
# Re-split by y_true to compute fold aggregates for consistency.
X_dummy = np.zeros((len(df), 1), dtype=float)
y_split = df["y_true"].values.astype(int)
y_pred_all = df["y_pred"].values.astype(int)

skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True,
                      random_state=RANDOM_STATE)

fold_macro = {
    "auc_ovr": [],  # NaN (no probabilities)
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

for fold_idx, (_, te_idx) in enumerate(skf.split(X_dummy, y_split), start=1):
    yte = y_split[te_idx]
    yhat = y_pred_all[te_idx]

    fm = fold_metrics_from_hard_predictions(
        y_true=yte, y_pred=yhat, class_labels=class_labels)

    for k in fold_macro:
        fold_macro[k].append(fm[k])

    fold_per_class["precision"].append(fm["per_class_precision"])
    fold_per_class["recall"].append(fm["per_class_recall"])
    fold_per_class["specificity"].append(fm["per_class_specificity"])
    fold_per_class["f1"].append(fm["per_class_f1"])
    fold_per_class["support"].append(fm["per_class_support"])

macro_rows = []
for metric_name, values in fold_macro.items():
    mu, sd = mean_std(np.array(values, dtype=float))
    macro_rows.append({"metric": metric_name, "mean": mu, "std": sd})

macro_df = pd.DataFrame(macro_rows)
macro_csv = os.path.join(OUT_DIR, "step4_min_fusion_svm_cv_macro_metrics.csv")
macro_df.to_csv(macro_csv, index=False)

# %% 3) "CV-style" per-class metrics (mean ± std across folds)
pc_prec = np.vstack(fold_per_class["precision"])
pc_rec = np.vstack(fold_per_class["recall"])
pc_spec = np.vstack(fold_per_class["specificity"])
pc_f1 = np.vstack(fold_per_class["f1"])

pc_support_total = np.sum(
    np.vstack(fold_per_class["support"]), axis=0).astype(int)

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
per_class_csv = os.path.join(
    OUT_DIR, "step4_min_fusion_svm_cv_per_class_metrics.csv")
per_class_df.to_csv(per_class_csv, index=False)

# %% 4) Per-sample fused predictions CSV
# Required structure:
# [ID, Auscultation_Point, ecg_predicted_label, pcg_predicted_label, mSQA_min, manual_label, predicted_label]
# manual_label == mSQA_min
df_out = df[[
    "ID",
    "Auscultation_Point",
    "ecg_predicted_label",
    "pcg_predicted_label",
    "mSQA_min",
]].copy()

df_out["manual_label"] = df_out["mSQA_min"]
df_out["predicted_label"] = df["y_pred"].map(INT_TO_CLASS)

df_out = df_out[[
    "ID",
    "Auscultation_Point",
    "ecg_predicted_label",
    "pcg_predicted_label",
    "mSQA_min",
    "manual_label",
    "predicted_label",
]]

pred_csv = os.path.join(OUT_DIR, "step4_min_fusion_svm_oof_predictions.csv")
df_out.to_csv(pred_csv, index=False)

# %% Minimal console summary
print(f"[Step 4 - SVM] Saved outputs to: {OUT_DIR}")
print(" - Confusion matrix (% PNG):    step4_min_fusion_svm_confusion_matrix_percent.png")
print(" - CV macro metrics:            step4_min_fusion_svm_cv_macro_metrics.csv")
print(" - CV per-class metrics:        step4_min_fusion_svm_cv_per_class_metrics.csv")
print(" - OOF predictions (per-row):   step4_min_fusion_svm_oof_predictions.csv")
