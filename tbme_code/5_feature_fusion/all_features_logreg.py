# -*- coding: utf-8 -*-
"""
All features fusion (PCG SQIs + ECG SQIs + alignment_metric*) + 3-class Logistic Regression (OOF-CV)
+ (optional) Feature significance analysis

File
----
Multimodal_SQA/tbme_code/5_feature_fusion/all_features_logreg.py

Purpose
-------
Train and evaluate a 3-class Logistic Regression classifier using:
  - PCG unimodal SQIs (Step 2 extracted CSV)
  - ECG unimodal SQIs (Step 3 extracted CSV)
  - Multimodal alignment metrics (ulsge_quality_metrics.pkl: columns alignment_metric*)

Ground truth: manual mSQA_min from ulsge_manual_sqa.xlsx.

Manual 6-point scale mapped into 3 classes:
    0           -> low_quality
    1, 2        -> uncertain
    3, 4, 5     -> high_quality

Evaluation protocol
-------------------
All reported metrics and per-sample predictions are computed from OUT-OF-FOLD (OOF)
predictions using StratifiedKFold cross-validation.

Artifacts generated (all OOF-based)
-----------------------------------
1) Confusion matrix (row-normalized %) PNG.
2) CV macro metrics (mean ± std) CSV.
3) CV per-class metrics (mean ± std) CSV.
4) Per-sample OOF predictions CSV:
   [ID, Auscultation_Point, ...fused_features..., mSQA_min, manual_label, predicted_label]

Feature significance analysis (additional)
------------------------------------------
A) Coefficient-based importance from a final model fit on ALL data:
   - absolute standardized coefficient magnitude (per class and aggregated)
   - saved as CSV
B) Permutation importance (macro_f1 scoring) on full dataset (not CV-aware; complementary):
   - saved as CSV
C) Optional p-values for coefficients via statsmodels MNLogit:
   - To avoid numerical warnings/instability, we use a stabilized MNLogit:
     near-constant drop + high-corr drop + VIF pruning + RobustScaler + clipping.
   - If fit is unstable (non-finite params/pvalues), we SKIP exporting p-values.

Fixes included
--------------
1) sklearn FutureWarning about `multi_class`:
   - Removed `multi_class` from LogisticRegression (default will be multinomial in future).
   - roc_auc_score multi_class="ovr" remains correct (that's not the warning).
2) statsmodels overflow/divide warnings:
   - Stabilized MNLogit fitting (see C above), and skip export if unstable.

@author: Daniel Proaño-Guevara
"""

# %% Imports
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
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
    from sklearn.inspection import permutation_importance
except Exception as e:
    raise ImportError(
        "scikit-learn is required for this script. "
        f"Install it and rerun. Root error: {e}"
    )

# Keep console clean from generic sklearn user warnings
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

# alignment metrics pickle (relative: tbme_code/5_feature_fusion -> ../../../ulsge_quality_metrics.pkl)
ALIGN_PKL_PATH = os.path.join(THIS_DIR, "..", "..", "..", "ulsge_quality_metrics.pkl")

# manual excel (same known location)
MQ_PATH = os.path.join(THIS_DIR, "..", "..", "..", "ulsge_manual_sqa.xlsx")

OUT_DIR = "exp_step5_all_features_logreg_3class"
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

PCG_SQI_COLS = [
    "seSQI", "cpSQI", "pr100_200SQI", "pr200_400SQI",
    "mean_133_267", "median_133_267", "max_600_733",
    "diff_peak_sqi", "svdSQI"
]
ECG_SQI_COLS = ["bSQI", "pSQI", "sSQI", "kSQI", "fSQI", "basSQI"]

# Significance toggles
RUN_COEF_IMPORTANCE = True
RUN_PERMUTATION_IMPORTANCE = True

# Recommended default: keep p-values optional, stabilized, and skip safely if unstable.
RUN_STATSMODELS_PVALUES = True


# %% Utilities
def _norm_point(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("_", "", regex=False).str.upper()


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

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=display_labels)
    disp.plot(ax=ax, values_format=".1f", cmap="Blues", colorbar=True)
    ax.set_title("All features — LogReg — OOF Confusion Matrix (%)")
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


def drop_near_constant(Xdf: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """Drop near-constant numeric columns (variance <= eps)."""
    if Xdf.empty:
        return Xdf
    vari = Xdf.var(axis=0, numeric_only=True)
    keep = vari[vari > eps].index
    return Xdf.loc[:, keep].copy()


def drop_highly_correlated(Xdf: pd.DataFrame, thresh: float = 0.98) -> pd.DataFrame:
    """Drop one of any pair of features with |corr| >= thresh (simple greedy)."""
    if Xdf.shape[1] <= 1:
        return Xdf
    corr = Xdf.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] >= thresh)]
    return Xdf.drop(columns=to_drop).copy()


def merge_pcg_ecg_features(pcg_df: pd.DataFrame, ecg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Step2 PCG features and Step3 ECG features by (ID, normalized point).
    Keeps PCG's Auscultation_Point string as authoritative.
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

    pcg_cols = [c for c in PCG_SQI_COLS if c in pcg.columns]
    ecg_cols = [c for c in ECG_SQI_COLS if c in ecg.columns]
    if not pcg_cols:
        raise ValueError(f"No expected PCG SQI cols found. Expected any of: {PCG_SQI_COLS}")
    if not ecg_cols:
        raise ValueError(f"No expected ECG SQI cols found. Expected any of: {ECG_SQI_COLS}")

    pcg_ren = {c: f"pcg_{c}" for c in pcg_cols}
    ecg_ren = {c: f"ecg_{c}" for c in ecg_cols}

    pcg = pcg[["ID", "Auscultation_Point", "_k_point"] + pcg_cols].rename(columns=pcg_ren)
    ecg = ecg[["ID", "Auscultation_Point", "_k_point"] + ecg_cols].rename(columns=ecg_ren)

    merged = pd.merge(
        pcg,
        ecg.drop(columns=["Auscultation_Point"]),  # avoid duplicate point string
        on=["ID", "_k_point"],
        how="inner",
    )
    return merged.drop(columns=["_k_point"])


def merge_alignment_metrics(base_df: pd.DataFrame, align_df: pd.DataFrame) -> pd.DataFrame:
    """Merge alignment_metric* columns into base_df by (ID, normalized point)."""
    a = align_df.copy()
    b = base_df.copy()

    for c in ["ID", "Auscultation_Point"]:
        if c not in a.columns:
            raise ValueError(f"Alignment PKL missing '{c}'. Found: {list(a.columns)}")
        if c not in b.columns:
            raise ValueError(f"Base DF missing '{c}'. Found: {list(b.columns)}")

    a["ID"] = a["ID"].astype(str)
    b["ID"] = b["ID"].astype(str)

    a["_k_point"] = _norm_point(a["Auscultation_Point"])
    b["_k_point"] = _norm_point(b["Auscultation_Point"])

    align_cols = [c for c in a.columns if str(c).startswith("alignment_metric")]
    if not align_cols:
        raise ValueError("No alignment_metric* columns found in ulsge_quality_metrics.pkl")

    a = a[["ID", "_k_point"] + align_cols].copy()

    out = pd.merge(
        b,
        a,
        on=["ID", "_k_point"],
        how="inner",
    )
    return out.drop(columns=["_k_point"])


def merge_with_manual_msqa(allfeat_df: pd.DataFrame, manual_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all features with manual mSQA_min using project join rule:
      - allfeat.ID == manual.Trial
      - allfeat.Auscultation_Point == manual.Spot after normalization
    """
    f = allfeat_df.copy()
    m = manual_df.copy()

    for c in ["Trial", "Spot", "mSQA_min"]:
        if c not in m.columns:
            raise ValueError(
                f"Manual Excel missing required column '{c}'. Found: {list(m.columns)}"
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


# %% Load inputs
print("[All features] Loading PCG SQIs...")
pcg_feat = pd.read_csv(PCG_FEAT_PATH)

print("[All features] Loading ECG SQIs...")
ecg_feat = pd.read_csv(ECG_FEAT_PATH)

print("[All features] Loading alignment metrics PKL...")
align_df = pd.read_pickle(ALIGN_PKL_PATH)

print("[All features] Loading manual labels Excel...")
m_quality = pd.read_excel(MQ_PATH)

# %% Build fused feature table: ECG+PCG then add alignment metrics
fused_unimodal = merge_pcg_ecg_features(pcg_feat, ecg_feat)
print(f"[All features] Joined ECG+PCG rows: {len(fused_unimodal)}")

fused_all = merge_alignment_metrics(fused_unimodal, align_df)
print(f"[All features] Added alignment metrics rows: {len(fused_all)}")

merged = merge_with_manual_msqa(fused_all, m_quality)
print(f"[All features] Joined with manual mSQA_min rows: {len(merged)}")

# %% Label preprocessing
df = merged.copy()

df["mSQA_min"] = pd.to_numeric(df["mSQA_min"], errors="coerce")
df = df.dropna(subset=["mSQA_min"]).copy()

df["manual_label"] = df["mSQA_min"].astype(int).map(RELABEL_MAP)
df = df.dropna(subset=["manual_label"]).copy()

le = LabelEncoder()
le.fit(CLASS_ORDER)
df["y"] = le.transform(df["manual_label"].values).astype(int)

# Feature columns: everything except metadata/labels
non_feature_cols = {"ID", "Auscultation_Point", "mSQA_min", "manual_label", "y"}
feature_cols = [c for c in df.columns if c not in non_feature_cols]

# Strict completeness
df = df.dropna(subset=feature_cols).copy()

X = df[feature_cols].values
y = df["y"].values.astype(int)

df["ID"] = df["ID"].astype(str)
df["Auscultation_Point"] = df["Auscultation_Point"].astype(str)

class_labels = np.array([0, 1, 2], dtype=int)
class_names = CLASS_ORDER

# %% Model pipeline (FIX: removed multi_class to avoid FutureWarning)
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    ))
])

# %% Cross-validated training (OOF)
skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

n = len(df)
oof_pred = np.full(shape=(n,), fill_value=-1, dtype=int)
oof_proba = np.full(shape=(n, len(class_labels)), fill_value=np.nan, dtype=float)

fold_macro = {"auc_ovr": [], "accuracy": [], "macro_sensitivity": [], "macro_specificity": [], "macro_f1": []}
fold_per_class = {"precision": [], "recall": [], "specificity": [], "f1": [], "support": []}

for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    lr_pipe.fit(Xtr, ytr)
    yhat = lr_pipe.predict(Xte)
    yproba = lr_pipe.predict_proba(Xte)

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
    out_prefix="all_features_logreg_cv",
)

# %% 2) CV macro metrics CSV
macro_rows = []
for metric_name, values in fold_macro.items():
    mu, sd = mean_std(np.array(values, dtype=float))
    macro_rows.append({"metric": metric_name, "mean": mu, "std": sd})

macro_df = pd.DataFrame(macro_rows)
macro_csv = os.path.join(OUT_DIR, "all_features_logreg_cv_macro_metrics.csv")
macro_df.to_csv(macro_csv, index=False)

# %% 3) CV per-class metrics CSV
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
per_class_csv = os.path.join(OUT_DIR, "all_features_logreg_cv_per_class_metrics.csv")
per_class_df.to_csv(per_class_csv, index=False)

# %% 4) Per-sample OOF predictions CSV
pred_df = df[["ID", "Auscultation_Point"] + feature_cols + ["mSQA_min", "manual_label"]].copy()
pred_df["predicted_label"] = np.array(class_names, dtype=object)[oof_pred]
pred_df = pred_df[["ID", "Auscultation_Point"] + feature_cols + ["mSQA_min", "manual_label", "predicted_label"]]

pred_csv = os.path.join(OUT_DIR, "all_features_logreg_oof_predictions.csv")
pred_df.to_csv(pred_csv, index=False)

# %% -------------------------
# %% Feature significance analysis
# %% -------------------------
print("[All features] Feature significance analysis...")

# Fit final model on ALL data (for coefficient inspection)
lr_pipe.fit(X, y)
scaler = lr_pipe.named_steps["scaler"]
lr = lr_pipe.named_steps["lr"]

if RUN_COEF_IMPORTANCE:
    coef = lr.coef_
    if coef.ndim != 2 or coef.shape[1] != len(feature_cols):
        raise RuntimeError("Unexpected coefficient shape from LogisticRegression.")

    coef_df = pd.DataFrame(coef, columns=feature_cols, index=[f"class_{c}" for c in class_names])

    agg_importance = np.mean(np.abs(coef), axis=0)
    coef_importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_coef_across_classes": agg_importance.astype(float),
    }).sort_values("mean_abs_coef_across_classes", ascending=False)

    coef_long_rows = []
    for ci, cname in enumerate(class_names):
        for fi, fname in enumerate(feature_cols):
            coef_long_rows.append({
                "class": cname,
                "feature": fname,
                "coef": float(coef[ci, fi]),
                "abs_coef": float(abs(coef[ci, fi])),
            })
    coef_long_df = pd.DataFrame(coef_long_rows).sort_values(["class", "abs_coef"], ascending=[True, False])

    coef_importance_csv = os.path.join(OUT_DIR, "all_features_logreg_coef_importance.csv")
    coef_long_csv = os.path.join(OUT_DIR, "all_features_logreg_coef_by_class.csv")
    coef_importance_df.to_csv(coef_importance_csv, index=False)
    coef_long_df.to_csv(coef_long_csv, index=False)

if RUN_PERMUTATION_IMPORTANCE:
    perm = permutation_importance(
        lr_pipe,
        X,
        y,
        n_repeats=25,
        random_state=RANDOM_STATE,
        scoring="f1_macro",
    )
    perm_df = pd.DataFrame({
        "feature": feature_cols,
        "perm_importance_mean_f1_macro": perm.importances_mean.astype(float),
        "perm_importance_std_f1_macro": perm.importances_std.astype(float),
    }).sort_values("perm_importance_mean_f1_macro", ascending=False)

    perm_csv = os.path.join(OUT_DIR, "all_features_logreg_permutation_importance_f1_macro.csv")
    perm_df.to_csv(perm_csv, index=False)

# %% Optional: statsmodels p-values (stabilized) — may skip if unstable
if RUN_STATSMODELS_PVALUES:
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from sklearn.preprocessing import RobustScaler

        def _compute_vif_df(X_: pd.DataFrame) -> pd.DataFrame:
            Xv = X_.values.astype(float)
            rows = []
            for i, c in enumerate(X_.columns):
                rows.append({"feature": c, "vif": float(variance_inflation_factor(Xv, i))})
            return pd.DataFrame(rows).sort_values("vif", ascending=False)

        def _prune_high_vif(X_: pd.DataFrame, vif_thresh: float = 50.0, max_drop: int = 50) -> pd.DataFrame:
            Xp = X_.copy()
            dropped = 0
            while True:
                if Xp.shape[1] <= 2:
                    break
                vif_df = _compute_vif_df(Xp)
                worst = vif_df.iloc[0]
                if worst["vif"] <= vif_thresh:
                    break
                Xp = Xp.drop(columns=[worst["feature"]])
                dropped += 1
                if dropped >= max_drop:
                    break
            return Xp

        # Build statsmodels-only design matrix (does not change sklearn outputs)
        Xdf = df[feature_cols].copy()

        # Stabilize: constant/corr/VIF pruning
        Xdf = drop_near_constant(Xdf, eps=1e-12)
        Xdf = drop_highly_correlated(Xdf, thresh=0.98)
        Xdf = _prune_high_vif(Xdf, vif_thresh=50.0, max_drop=50)

        sm_feature_cols = list(Xdf.columns)

        # Robust scaling + clipping to prevent exp overflow
        Xz = RobustScaler(quantile_range=(25.0, 75.0)).fit_transform(Xdf.values.astype(float))
        Xz = np.clip(Xz, -10.0, 10.0)
        Xz = sm.add_constant(Xz, prepend=True, has_constant="add")

        mn = sm.MNLogit(y, Xz)

        # Silence runtime warnings during fit; validate outputs explicitly
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            res = mn.fit(method="bfgs", maxiter=800, disp=False)

        params = res.params
        pvals = res.pvalues

        if not np.isfinite(params.to_numpy()).all() or not np.isfinite(pvals.to_numpy()).all():
            raise RuntimeError("MNLogit produced non-finite params/p-values (unstable fit).")

        feat_plus = ["const"] + sm_feature_cols

        rows = []
        for col_idx in range(params.shape[1]):  # classes vs baseline
            cls = f"vs_baseline_{col_idx}"
            for r_idx, fname in enumerate(feat_plus):
                rows.append({
                    "comparison": cls,
                    "feature": fname,
                    "coef": float(params.iloc[r_idx, col_idx]),
                    "p_value": float(pvals.iloc[r_idx, col_idx]),
                })

        sm_df = pd.DataFrame(rows).sort_values(["comparison", "p_value"], ascending=[True, True])
        sm_csv = os.path.join(OUT_DIR, "all_features_logreg_statsmodels_pvalues.csv")
        sm_df.to_csv(sm_csv, index=False)
        print(f"[All features] statsmodels p-values saved: {sm_csv}")

    except Exception as e:
        print(f"[All features] statsmodels p-values skipped: {e}")

# %% Minimal console summary
print(f"[All features] Saved outputs to: {OUT_DIR}")
print(" - Confusion matrix (% PNG):    all_features_logreg_cv_confusion_matrix_percent.png")
print(" - CV macro metrics:            all_features_logreg_cv_macro_metrics.csv")
print(" - CV per-class metrics:        all_features_logreg_cv_per_class_metrics.csv")
print(" - OOF predictions (per-row):   all_features_logreg_oof_predictions.csv")
if RUN_COEF_IMPORTANCE:
    print(" - Coef importance:             all_features_logreg_coef_importance.csv")
    print(" - Coefs by class:              all_features_logreg_coef_by_class.csv")
if RUN_PERMUTATION_IMPORTANCE:
    print(" - Permutation importance:      all_features_logreg_permutation_importance_f1_macro.csv")
if RUN_STATSMODELS_PVALUES:
    print(" - Statsmodels p-values:        all_features_logreg_statsmodels_pvalues.csv (if stable; otherwise skipped)")
