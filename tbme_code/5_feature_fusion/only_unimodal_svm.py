# -*- coding: utf-8 -*-
"""
Feature fusion (ECG SQIs + PCG SQIs) + 3-class RBF-SVM (OOF-CV)

File
----
Multimodal_SQA/tbme_code/5_feature_fusion/only_unimodal_svm.py

Purpose
-------
Train and evaluate a 3-class SVM classifier (RBF kernel) using *feature fusion* of
unimodal SQIs extracted from:
  - PCG (Step 2 extractor CSV)
  - ECG (Step 3 extractor CSV)

Ground truth is the multimodal manual label mSQA_min from ulsge_manual_sqa.xlsx.

The manual 6-point scale is mapped into 3 classes:
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
   [ID, Auscultation_Point, ...fused_features..., mSQA_min, manual_label, predicted_label]

Notes
-----
- Feature alignment is done by (project convention):
    PCG/ECG feature row keys:
        ID (as string)
        Auscultation_Point (normalized: remove '_' and uppercase)
  - Manual label join:
        features.ID == manual.Trial (as string)
        features.Auscultation_Point == manual.Spot (normalized)

Inputs (relative to this script)
--------------------------------
1) PCG SQIs CSV:
   ../2_unimodal_pcg/exp_step2_pcg_features/step2_pcg_unimodal_features_extracted.csv
2) ECG SQIs CSV:
   ../3_unimodal_ecg/exp_step3_ecg_features/step3_ecg_unimodal_features_extracted.csv
3) Manual labels Excel:
   ../../../ulsge_manual_sqa.xlsx

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

# %% Global config
THIS_DIR = os.path.dirname(__file__)

PCG_FEAT_PATH = os.path.join(
    THIS_DIR,
    "..",
    "2_unimodal_pcg",
    "exp_step2_pcg_features",
    "step2_pcg_unimodal_features_extracted.csv",
)

ECG_FEAT_PATH = os.path.join(
    THIS_DIR,
    "..",
    "3_unimodal_ecg",
    "exp_step3_ecg_features",
    "step3_ecg_unimodal_features_extracted.csv",
)

MQ_PATH = os.path.join(THIS_DIR, "..", "..", "..", "ulsge_manual_sqa.xlsx")

OUT_DIR = "exp_step5_feature_fusion_only_unimodal_svm_3class"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS_CV = 5

RELABEL_MAP = {
    0: "low_quality",
    1: "uncertain",
    2: "uncertain",
    3: "high_quality",
    4: "high_quality",
    5: "high_quality",
}

CLASS_ORDER = ["low_quality", "uncertain", "high_quality"]

# Expected feature columns (stable model inputs)
PCG_SQI_COLS = [
    "seSQI", "cpSQI", "pr100_200SQI", "pr200_400SQI",
    "mean_133_267", "median_133_267", "max_600_733",
    "diff_peak_sqi", "svdSQI"
]

ECG_SQI_COLS = [
    "bSQI",
    "pSQI",
    "sSQI",
    "kSQI",
    "fSQI",
    "basSQI",
]


# %% Utilities

def _norm_point(series: pd.Series) -> pd.Series:
    """Normalize auscultation points: remove '_' and uppercase."""
    return series.astype(str).str.replace("_", "", regex=False).str.upper()


def merge_ecg_pcg_features(pcg_df: pd.DataFrame, ecg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge PCG and ECG feature tables by:
      - ID (string)
      - normalized Auscultation_Point

    Output columns:
      - ID, Auscultation_Point
      - fused feature set:
          pcg_* (prefixed) + ecg_* (prefixed)
    """
    pcg = pcg_df.copy()
    ecg = ecg_df.copy()

    for c in ["ID", "Auscultation_Point"]:
        if c not in pcg.columns:
            raise ValueError(f"PCG feature file missing '{c}'. Found: {list(pcg.columns)}")
        if c not in ecg.columns:
            raise ValueError(f"ECG feature file missing '{c}'. Found: {list(ecg.columns)}")

    pcg["ID"] = pcg["ID"].astype(str)
    ecg["ID"] = ecg["ID"].astype(str)

    pcg["_k_point"] = _norm_point(pcg["Auscultation_Point"])
    ecg["_k_point"] = _norm_point(ecg["Auscultation_Point"])

    pcg_feat_cols = [c for c in PCG_SQI_COLS if c in pcg.columns]
    ecg_feat_cols = [c for c in ECG_SQI_COLS if c in ecg.columns]

    if not pcg_feat_cols:
        raise ValueError(f"No expected PCG SQI cols found. Expected any of: {PCG_SQI_COLS}")
    if not ecg_feat_cols:
        raise ValueError(f"No expected ECG SQI cols found. Expected any of: {ECG_SQI_COLS}")

    pcg_ren = {c: f"pcg_{c}" for c in pcg_feat_cols}
    ecg_ren = {c: f"ecg_{c}" for c in ecg_feat_cols}

    pcg = pcg[["ID", "Auscultation_Point", "_k_point"] + pcg_feat_cols].rename(columns=pcg_ren)
    ecg = ecg[["ID", "Auscultation_Point", "_k_point"] + ecg_feat_cols].rename(columns=ecg_ren)

    merged = pd.merge(
        pcg,
        ecg.drop(columns=["Auscultation_Point"]),  # keep PCG string as deterministic choice
        on=["ID", "_k_point"],
        how="inner",
    )

    merged = merged.drop(columns=["_k_point"])
    return merged


def merge_with_manual_msqa(fused_df: pd.DataFrame, manual_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fused features with manual mSQA_min labels using project convention:
      - fused.ID == manual.Trial (as str)
      - fused.Auscultation_Point == manual.Spot after normalization

    Returns:
      - ID, Auscultation_Point
      - mSQA_min
      - fused features...
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
    cols = ["ID", "Auscultation_Point", "mSQA_min"] + [
        c for c in out.columns if c not in ["ID", "Auscultation_Point", "mSQA_min"]
    ]
    return out[cols].copy()


def compute_specificity_from_cm(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    with np.errstate(divide="ignore", invalid="ignore"):
        spec = np.where((tn + fp) > 0, tn / (tn + fp), 0.0)
    return spec


def save_confusion_matrix_artifacts(y_true, y_pred, labels, display_labels, out_prefix: str) -> None:
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_percent = np.where(row_sums > 0, cm_counts.astype(float) / row_sums * 100.0, 0.0)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=display_labels)
    disp.plot(ax=ax, values_format=".1f", cmap="Blues", colorbar=True)
    ax.set_title("Step 5 — Feature fusion (ECG+PCG SQIs) — RBF-SVM OOF Confusion Matrix (%)")
    plt.tight_layout()

    png_path = os.path.join(OUT_DIR, f"{out_prefix}_confusion_matrix_percent.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fold_metrics_from_predictions(y_true, y_pred, y_proba, class_labels) -> dict:
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
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x)) if x.size else 0.0
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return mu, sd


# %% Load inputs
pcg_feat = pd.read_csv(PCG_FEAT_PATH)
ecg_feat = pd.read_csv(ECG_FEAT_PATH)
m_quality = pd.read_excel(MQ_PATH)

# %% Merge ECG+PCG features (feature fusion)
fused = merge_ecg_pcg_features(pcg_feat, ecg_feat)
print(f"[Step 5] Joined ECG+PCG feature rows: {len(fused)}")

# %% Merge with manual labels (mSQA_min)
merged = merge_with_manual_msqa(fused, m_quality)
print(f"[Step 5] Joined with manual mSQA_min rows: {len(merged)}")

# %% Label preprocessing (mSQA_min -> 3-class y)
df = merged.copy()

df["mSQA_min"] = pd.to_numeric(df["mSQA_min"], errors="coerce")
df = df.dropna(subset=["mSQA_min"]).copy()

df["gt_3class_str"] = df["mSQA_min"].astype(int).map(RELABEL_MAP)
df = df.dropna(subset=["gt_3class_str"]).copy()

df["manual_label"] = df["gt_3class_str"].astype(str)

le = LabelEncoder()
le.fit(CLASS_ORDER)
df["y"] = le.transform(df["manual_label"].values).astype(int)

non_feature_cols = {"ID", "Auscultation_Point", "mSQA_min", "gt_3class_str", "manual_label", "y"}
feature_cols = [c for c in df.columns if c not in non_feature_cols]

# Strict completeness
df = df.dropna(subset=feature_cols).copy()

X = df[feature_cols].values
y = df["y"].values.astype(int)

df["ID"] = df["ID"].astype(str)
df["Auscultation_Point"] = df["Auscultation_Point"].astype(str)

class_labels = np.array([0, 1, 2], dtype=int)
class_names = CLASS_ORDER

# %% Model pipeline (RBF SVM)
# SVM with RBF kernel is sensitive to scale -> StandardScaler.
# probability=True enables predict_proba() for multiclass ROC-AUC.
# class_weight='balanced' compensates for imbalance.
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
skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

n = len(df)
oof_pred = np.full(shape=(n,), fill_value=-1, dtype=int)
oof_proba = np.full(shape=(n, len(class_labels)), fill_value=np.nan, dtype=float)

fold_macro = {"auc_ovr": [], "accuracy": [], "macro_sensitivity": [], "macro_specificity": [], "macro_f1": []}
fold_per_class = {"precision": [], "recall": [], "specificity": [], "f1": [], "support": []}

for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    svm_pipe.fit(Xtr, ytr)
    yhat = svm_pipe.predict(Xte)
    yproba = svm_pipe.predict_proba(Xte)

    oof_pred[te_idx] = yhat
    oof_proba[te_idx, :] = yproba

    fm = fold_metrics_from_predictions(y_true=yte, y_pred=yhat, y_proba=yproba, class_labels=class_labels)

    for k in fold_macro:
        fold_macro[k].append(fm[k])

    fold_per_class["precision"].append(fm["per_class_precision"])
    fold_per_class["recall"].append(fm["per_class_recall"])
    fold_per_class["specificity"].append(fm["per_class_specificity"])
    fold_per_class["f1"].append(fm["per_class_f1"])
    fold_per_class["support"].append(fm["per_class_support"])

if np.any(oof_pred < 0) or np.any(~np.isfinite(oof_proba)):
    raise RuntimeError("OOF predictions/probabilities are incomplete. Check CV split loop.")

# %% 1) Confusion matrix (OOF)
save_confusion_matrix_artifacts(
    y_true=y,
    y_pred=oof_pred,
    labels=class_labels,
    display_labels=class_names,
    out_prefix="step5_feature_fusion_svm_cv",
)

# %% 2) CV macro metrics (mean ± std)
macro_rows = []
for metric_name, values in fold_macro.items():
    mu, sd = mean_std(np.array(values, dtype=float))
    macro_rows.append({"metric": metric_name, "mean": mu, "std": sd})

macro_df = pd.DataFrame(macro_rows)
macro_csv = os.path.join(OUT_DIR, "step5_feature_fusion_svm_cv_macro_metrics.csv")
macro_df.to_csv(macro_csv, index=False)

# %% 3) CV per-class metrics (mean ± std)
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
per_class_csv = os.path.join(OUT_DIR, "step5_feature_fusion_svm_cv_per_class_metrics.csv")
per_class_df.to_csv(per_class_csv, index=False)

# %% 4) Per-sample OOF predictions CSV
pred_df = df[["ID", "Auscultation_Point"] + feature_cols + ["mSQA_min", "manual_label"]].copy()
pred_df["predicted_label"] = np.array(class_names, dtype=object)[oof_pred]

pred_df = pred_df[["ID", "Auscultation_Point"] + feature_cols + ["mSQA_min", "manual_label", "predicted_label"]]
pred_csv = os.path.join(OUT_DIR, "step5_feature_fusion_svm_oof_predictions.csv")
pred_df.to_csv(pred_csv, index=False)

# %% Minimal console summary
print(f"[Step 5] Saved outputs to: {OUT_DIR}")
print(" - Confusion matrix (% PNG):    step5_feature_fusion_svm_cv_confusion_matrix_percent.png")
print(" - CV macro metrics:            step5_feature_fusion_svm_cv_macro_metrics.csv")
print(" - CV per-class metrics:        step5_feature_fusion_svm_cv_per_class_metrics.csv")
print(" - OOF predictions (per-row):   step5_feature_fusion_svm_oof_predictions.csv")
