# -*- coding: utf-8 -*-
"""
STEP 1 + STEP 2 — 3-class experiments (LogReg)
Spyder/Jupyter-friendly, sequential execution.
Step 1:
- Features: alignment_metric* (from ulsge_quality_metrics.pkl)
- Target: manual SQA from m_quality column E (or 'mSQA_min' if present), relabeled to 3-class

Step 2:
- Features: ALL unimodal PCG SQIs (computed from pcg_ulsge.pkl after bandpass filtering)
- Target: manual PCG score from Excel column D named 'PCG', relabeled to 3-class

Common outputs (each step):
- Test split metrics: AUC OvR, Accuracy, Macro Sensitivity, Macro Specificity, Macro F1
- Per-class metrics (precision/recall/specificity/F1/support)
- Confusion matrix (%) for the test split (saved)
- Cross-validated summary metrics (StratifiedKFold)

Designed so you can append Step 3+ at the bottom without touching utilities.
"""

# %% Imports
import os
import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing_lib as pplib
import sqi_pcg_lib
import sqi_ecg_lib

try:
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        confusion_matrix,
        ConfusionMatrixDisplay,
        roc_auc_score,
        accuracy_score,
        f1_score,
        recall_score,
    )
except Exception as e:
    raise ImportError(
        "scikit-learn is required for this script. "
        f"Install it and rerun. Root error: {e}"
    )

try:
    from scipy.stats import ttest_rel
except Exception as e:
    warnings.warn(
        f"[WARN] scipy not available ({e}). Step 6 will run but p-values cannot be computed reliably."
    )
    ttest_rel = None

# %% Global config (edit as needed)
AQ_PATH = r'..\ulsge_quality_metrics.pkl'
MQ_PATH = r'..\ulsge_manual_sqa.xlsx'

OUT_DIR = "exp_step1_logreg_3class"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.25
N_SPLITS_CV = 5

# --- Relabeling (must match sqa_svm) ---
# If sqa_svm uses a different scheme, change ONLY this dict.
RELABEL_MAP = {
    0: "low_quality",
    1: "uncertain",
    2: "uncertain",
    3: "high_quality",
    4: "high_quality",
    5: "high_quality",
}
# fixed display order for reports
CLASS_ORDER = ["low_quality", "uncertain", "high_quality"]

print("Configured:")
print(f"  AQ_PATH: {AQ_PATH}")
print(f"  MQ_PATH: {MQ_PATH}")
print(f"  OUT_DIR: {OUT_DIR}")
print(f"  TEST_SIZE: {TEST_SIZE}, N_SPLITS_CV: {N_SPLITS_CV}")
print(f"  CLASS_ORDER: {CLASS_ORDER}")

# %% =========================
# %% Utilities (reused by all steps)
# %% =========================

# %% Merge dataframes (project-style, minimal output)


def merge_quality_dataframes(ex1_quality: pd.DataFrame, m_quality: pd.DataFrame) -> pd.DataFrame:
    """
    Merge automatic metrics (ex1_quality) with manual labels (m_quality) using:
      - ID == Trial  (as str)
      - Auscultation_Point == Spot (ignoring underscores, case-insensitive)

    Returns a CLEAN dataframe with only:
      - ID, Auscultation_Point
      - manual labels: mSQA_min, ECG, PCG (if present in m_quality)
      - all ex1 columns starting with 'alignment_metric'
    """
    ex1 = ex1_quality.copy()
    m = m_quality.copy()

    # Keys as str
    ex1["ID"] = ex1["ID"].astype(str)
    m["Trial"] = m["Trial"].astype(str)

    # Temporary normalized keys (NOT kept in result)
    ex1["_k_point"] = ex1["Auscultation_Point"].astype(
        str).str.replace("_", "", regex=False).str.upper()
    m["_k_spot"] = m["Spot"].astype(str).str.replace(
        "_", "", regex=False).str.upper()

    # Columns to bring from manual side (only what we need)
    manual_keep = ["Trial", "_k_spot"]
    for c in ["mSQA_min", "ECG", "PCG"]:
        if c in m.columns:
            manual_keep.append(c)

    # Merge
    merged = pd.merge(
        ex1,
        m[manual_keep],
        left_on=["ID", "_k_point"],
        right_on=["Trial", "_k_spot"],
        how="inner"
    )

    # Keep only desired columns in the final output
    alignment_cols = [c for c in ex1.columns if str(
        c).startswith("alignment_metric")]
    base_cols = ["ID", "Auscultation_Point"] + \
        [c for c in ["mSQA_min", "ECG", "PCG"] if c in merged.columns]

    result = merged[base_cols + alignment_cols].copy()

    return result


def compute_specificity_from_cm(cm: np.ndarray) -> np.ndarray:
    """Compute per-class specificity from a confusion matrix."""
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    with np.errstate(divide="ignore", invalid="ignore"):
        spec = np.where((tn + fp) > 0, tn / (tn + fp), 0.0)
    return spec


def per_class_metrics_from_cm(cm: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """Per-class precision/recall/specificity/F1/support computed from the confusion matrix."""
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        recall = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        specificity = np.where((tn + fp) > 0, tn / (tn + fp), 0.0)
        f1 = np.where((precision + recall) > 0, 2 * precision *
                      recall / (precision + recall), 0.0)

    support = cm.sum(axis=1).astype(int)

    return pd.DataFrame({
        "class": class_names,
        "support": support,
        "precision": precision,
        "sensitivity_recall": recall,
        "specificity": specificity,
        "f1": f1,
    })


def save_confusion_matrix_percent(y_true, y_pred, labels, display_labels, title, out_png):
    """
    Save confusion matrix as row-normalized percentage plot.
    NOTE: row normalization assumes each true class has at least one sample.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Avoid divide-by-zero if a class is absent in y_true (shouldn't happen with stratified split, but safe)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_percent = np.where(row_sums > 0, cm.astype(
            float) / row_sums * 100.0, 0.0)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_percent, display_labels=display_labels)
    disp.plot(ax=ax, values_format=".1f", cmap="Blues", colorbar=True)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nConfusion matrix saved: {out_png}")
    print("Confusion matrix (%), rows=true, cols=pred:")
    print(np.round(cm_percent, 1))

    return cm


def evaluate_test_split(model, Xtr, Xte, ytr, yte, class_labels, class_names, task_name, out_prefix):
    """
    Fit model on train, evaluate on test, save confusion matrix + per-class CSV, return metrics dict.
    Uses global OUT_DIR for artifact paths (can be temporarily overridden per step).
    """
    model.fit(Xtr, ytr)

    y_pred = model.predict(Xte)

    # Some models may not expose predict_proba; LogReg does. Keep a clear error if not.
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "Model/pipeline must implement predict_proba to compute AUC OvR.")
    y_proba = model.predict_proba(Xte)

    # AUC OvR for multiclass
    y_bin = label_binarize(yte, classes=class_labels)
    auc_ovr = roc_auc_score(y_bin, y_proba, multi_class="ovr")

    acc = accuracy_score(yte, y_pred)
    f1_macro = f1_score(yte, y_pred, average="macro")
    sens_macro = recall_score(yte, y_pred, average="macro")

    cm = confusion_matrix(yte, y_pred, labels=class_labels)
    spec_macro = float(np.mean(compute_specificity_from_cm(cm)))

    print("\n" + "-" * 70)
    print(f"{task_name} — TEST SPLIT METRICS")
    print("-" * 70)
    print(f"AUC OvR:            {auc_ovr:.4f}")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Macro Sensitivity:  {sens_macro:.4f}")
    print(f"Macro Specificity:  {spec_macro:.4f}")
    print(f"Macro F1:           {f1_macro:.4f}")

    per_cls = per_class_metrics_from_cm(cm, class_names)
    print("\nPer-class metrics:")
    print(per_cls.to_string(index=False))

    cm_png = os.path.join(OUT_DIR, f"{out_prefix}_cm_percent.png")
    _ = save_confusion_matrix_percent(
        y_true=yte,
        y_pred=y_pred,
        labels=class_labels,
        display_labels=class_names,
        title=f"{task_name} — Confusion Matrix (%)",
        out_png=cm_png
    )

    per_cls_csv = os.path.join(OUT_DIR, f"{out_prefix}_per_class_metrics.csv")
    per_cls.to_csv(per_cls_csv, index=False)
    print(f"Per-class metrics saved: {per_cls_csv}")

    return {
        "auc_ovr": float(auc_ovr),
        "accuracy": float(acc),
        "macro_sensitivity": float(sens_macro),
        "macro_specificity": float(spec_macro),
        "macro_f1": float(f1_macro),
        "per_class_df": per_cls,
        "cm": cm,
    }


def cross_validated_summary(pipeline, X, y, class_labels, task_name):
    """StratifiedKFold CV summary metrics (no confusion matrices)."""
    skf = StratifiedKFold(n_splits=N_SPLITS_CV,
                          shuffle=True, random_state=RANDOM_STATE)

    aucs, accs, recs, f1s, specs = [], [], [], [], []

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        pipeline.fit(Xtr, ytr)
        yhat = pipeline.predict(Xte)

        if not hasattr(pipeline, "predict_proba"):
            raise AttributeError(
                "Pipeline must implement predict_proba to compute AUC OvR.")
        yproba = pipeline.predict_proba(Xte)

        y_bin = label_binarize(yte, classes=class_labels)
        auc = roc_auc_score(y_bin, yproba, multi_class="ovr")
        acc = accuracy_score(yte, yhat)
        rec = recall_score(yte, yhat, average="macro")
        f1m = f1_score(yte, yhat, average="macro")

        cm = confusion_matrix(yte, yhat, labels=class_labels)
        spec = float(np.mean(compute_specificity_from_cm(cm)))

        aucs.append(auc)
        accs.append(acc)
        recs.append(rec)
        f1s.append(f1m)
        specs.append(spec)

        print(
            f"  CV fold {fold}/{N_SPLITS_CV}: AUC={auc:.4f}, Acc={acc:.4f}, Rec={rec:.4f}, Spec={spec:.4f}, F1={f1m:.4f}")

    def mean_std(x):
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        return mu, sd

    auc_mu, auc_sd = mean_std(aucs)
    acc_mu, acc_sd = mean_std(accs)
    rec_mu, rec_sd = mean_std(recs)
    spec_mu, spec_sd = mean_std(specs)
    f1_mu, f1_sd = mean_std(f1s)

    print("\n" + "-" * 70)
    print(f"{task_name} — CROSS-VALIDATED SUMMARY (StratifiedKFold)")
    print("-" * 70)
    print(f"AUC OvR:           {auc_mu:.4f} ± {auc_sd:.4f}")
    print(f"Accuracy:          {acc_mu:.4f} ± {acc_sd:.4f}")
    print(f"Macro Sensitivity: {rec_mu:.4f} ± {rec_sd:.4f}")
    print(f"Macro Specificity: {spec_mu:.4f} ± {spec_sd:.4f}")
    print(f"Macro F1:          {f1_mu:.4f} ± {f1_sd:.4f}")

    return {
        "cv_auc_mean": auc_mu, "cv_auc_std": auc_sd,
        "cv_acc_mean": acc_mu, "cv_acc_std": acc_sd,
        "cv_rec_mean": rec_mu, "cv_rec_std": rec_sd,
        "cv_spec_mean": spec_mu, "cv_spec_std": spec_sd,
        "cv_f1_mean": f1_mu, "cv_f1_std": f1_sd,
    }


def build_3class_target_from_mquality(merged_df: pd.DataFrame, m_quality_cols: pd.Index) -> tuple[pd.DataFrame, str]:
    """
    Select raw target from column E (index 4) unless 'mSQA_min' exists,
    then map to 3-class labels using RELABEL_MAP and encode according to CLASS_ORDER.

    Returns:
      df: dataframe with added columns: y_3class_str, y_3class
      y_raw_col: name of the raw manual column used
    """
    # Prefer explicit column name if present (more robust than positional indexing)
    if "mSQA_min" in merged_df.columns:
        y_raw_col = "mSQA_min"
        print("\nRaw target: using 'mSQA_min' (found by name).")
    else:
        if len(m_quality_cols) < 5:
            raise ValueError(
                "m_quality must have at least 5 columns to access Excel column E.")
        colE_name = m_quality_cols[4]  # Excel column E -> index 4
        if colE_name not in merged_df.columns:
            raise ValueError(
                f"Column E name '{colE_name}' not found after merge.")
        y_raw_col = colE_name
        print(
            f"\nRaw target: using m_quality column E -> '{y_raw_col}' (by index).")

    df = merged_df.copy()

    # Coerce to numeric and drop missing
    df[y_raw_col] = pd.to_numeric(df[y_raw_col], errors="coerce")
    df = df.dropna(subset=[y_raw_col]).copy()

    # Map to 3 classes and drop unmapped values (e.g., outside 0..5)
    df["y_3class_str"] = df[y_raw_col].astype(int).map(RELABEL_MAP)
    df = df.dropna(subset=["y_3class_str"]).copy()

    # Encode deterministically in CLASS_ORDER
    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    df["y_3class"] = le.transform(df["y_3class_str"].values)

    return df, y_raw_col


# %% =========================
# %% STEP 1 — LogReg on alignment_metric*, 3-class
# %% =========================
print("\n[Step 1] Loading datasets...")
ex1_quality = pd.read_pickle(r'..\ulsge_quality_metrics.pkl')
m_quality = pd.read_excel(MQ_PATH)

print("\n[Step 1] Merging (clean output)...")
merged = merge_quality_dataframes(ex1_quality, m_quality)

# Optional filtering (your preference)
if "mSQA_min" in merged.columns:
    merged = merged.dropna(subset=["mSQA_min"])
if "ECG" in merged.columns:
    merged = merged.dropna(subset=["ECG"])
if "PCG" in merged.columns:
    merged = merged.dropna(subset=["PCG"])

print(f"  merged rows: {len(merged)}")
print(f"  merged columns: {list(merged.columns)}")

alignment_metrics = [c for c in merged.columns if str(
    c).startswith("alignment_metric")]
if not alignment_metrics:
    raise ValueError("No alignment_metric* columns found after merge.")

print("\n[Step 1] Building 3-class target from mSQA_min...")
if "mSQA_min" not in merged.columns:
    raise ValueError(
        "mSQA_min not found in merged. Your manual file should contain it.")

df1 = merged.copy()
df1["mSQA_min"] = pd.to_numeric(df1["mSQA_min"], errors="coerce")
df1 = df1.dropna(subset=["mSQA_min"]).copy()

df1["y_3class_str"] = df1["mSQA_min"].astype(int).map(RELABEL_MAP)
df1 = df1.dropna(subset=["y_3class_str"]).copy()

le = LabelEncoder()
le.fit(CLASS_ORDER)
df1["y_3class"] = le.transform(df1["y_3class_str"].values)

# Drop rows with missing features
df1 = df1.dropna(subset=alignment_metrics).copy()

X1 = df1[alignment_metrics].values
y1 = df1["y_3class"].values

print("[Step 1] 3-class distribution:")
print(df1["y_3class_str"].value_counts())

keys1 = df1[["ID", "Auscultation_Point"]].astype(
    str).agg("||".join, axis=1).values

# Model (same as before)
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        multi_class="multinomial",
        solver="lbfgs",
        random_state=RANDOM_STATE
    ))
])

print("\n[Step 1] Train/test evaluation...")
Xtr, Xte, ytr, yte, ktr, kte = train_test_split(
    X1, y1, keys1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y1
)

res1_test = evaluate_test_split(
    model=lr_pipe,
    Xtr=Xtr, Xte=Xte,
    ytr=ytr, yte=yte,
    class_labels=np.array([0, 1, 2], dtype=int),
    class_names=CLASS_ORDER,
    task_name="Step 1 — LogReg (alignment metrics) — 3-class",
    out_prefix="step1_logreg_3class"
)

res1_test["keys"] = kte
res1_test["y_true"] = yte

# If evaluate_test_split already returns these keys, keep the assignments below;
# otherwise apply the evaluate_test_split patch shown after Step 1.
res1_test["y_pred"] = res1_test.get("y_pred", None)
res1_test["y_proba"] = res1_test.get("y_proba", None)

print("\n[Step 1] Cross-validated summary...")
res1_cv = cross_validated_summary(
    pipeline=lr_pipe,
    X=X1, y=y1,
    class_labels=np.array([0, 1, 2], dtype=int),
    task_name="Step 1 — LogReg (alignment metrics) — 3-class"
)


# %% =========================
# %% STEP 2 — PCG unimodal SQIs (ALL) + 3-class LogReg
# %% Uses manual PCG label from `merged["PCG"]`
# %% =========================
OUT_DIR_STEP2 = "exp_step2_pcg_unimodal_allSQI_logreg_3class"
os.makedirs(OUT_DIR_STEP2, exist_ok=True)

PCG_PKL_PATH = r"..\DatasetCHVNGE\pcg_ulsge.pkl"
PCG_FS = 3000
PCG_BPF_ORDER = 4
PCG_BPF_FC = [50, 250]

SE_M = 2
SE_R = 0.0008
SVD_HR_RANGE_BPM = (70, 220)

PCG_ID_COL = "ID"
PCG_POINT_COL = "Auscultation_Point"
PCG_SIGNAL_COL = "PCG"  # signal must be this

FEATURE_COLS_STEP2 = [
    "seSQI", "cpSQI", "pr100_200SQI", "pr200_400SQI",
    "mean_133_267", "median_133_267", "max_600_733",
    "diff_peak_sqi", "svdSQI"
]

# --- Preconditions ---
if "merged" not in globals():
    raise RuntimeError(
        "Step 2 expects `merged` from Step 1 (clean merged output).")
if "PCG" not in merged.columns:
    raise ValueError(
        "Manual PCG column not found in merged. Ensure m_quality has 'PCG' column.")
if "pplib" not in globals():
    raise ImportError(
        "preprocessing_lib (pplib) not available. Fix imports to run preprocessing.")

# --- Load PCG signals ---
print("\n[Step 2] Loading PCG signals...")
pcg_df = pd.read_pickle(PCG_PKL_PATH).copy()
pcg_df[PCG_ID_COL] = pcg_df[PCG_ID_COL].astype(str)

# --- Build a clean manual-label df (rename label to avoid collision) ---
manual_pcg_df = merged[["ID", "Auscultation_Point", "PCG"]].copy()
manual_pcg_df["ID"] = manual_pcg_df["ID"].astype(str)
manual_pcg_df = manual_pcg_df.rename(
    columns={"PCG": "PCG_manual"})  # prevents PCG_x/PCG_y

# --- Temporary normalized keys just for the join ---
pcg_df["_k_point"] = pcg_df[PCG_POINT_COL].astype(
    str).str.replace("_", "", regex=False).str.upper()
manual_pcg_df["_k_point"] = manual_pcg_df["Auscultation_Point"].astype(
    str).str.replace("_", "", regex=False).str.upper()

print("[Step 2] Joining signals with manual PCG labels (clean, collision-safe)...")
df2 = pd.merge(
    pcg_df[[PCG_ID_COL, PCG_POINT_COL, "_k_point", PCG_SIGNAL_COL]].copy(),
    manual_pcg_df[["ID", "Auscultation_Point",
                   "_k_point", "PCG_manual"]].copy(),
    left_on=[PCG_ID_COL, "_k_point"],
    right_on=["ID", "_k_point"],
    how="inner"
)

# Drop the join helper
df2 = df2.drop(columns=["_k_point"])

# Drop redundant manual auscultation point (keep the signal-side one)
# This keeps df readable and avoids *_x/*_y patterns.
df2 = df2.drop(columns=["Auscultation_Point_y"]).rename(
    columns={"Auscultation_Point_x": "Auscultation_Point"})

print(f"  joined rows: {len(df2)}")
if "PCG_x" in df2.columns or "PCG_y" in df2.columns:
    raise RuntimeError(
        "Still got PCG_x/PCG_y. The rename to PCG_manual did not apply correctly.")

# --- Preprocess signal ---
print("\n[Step 2] Preprocessing PCG (bandpass)...")
df2[PCG_SIGNAL_COL] = df2[PCG_SIGNAL_COL].apply(
    lambda data: pplib.butterworth_filter(
        data,
        filter_topology="bandpass",
        order=PCG_BPF_ORDER,
        fs=PCG_FS,
        fc=PCG_BPF_FC
    )
)

# --- Build 3-class target from PCG_manual ---
df2["PCG_manual"] = pd.to_numeric(df2["PCG_manual"], errors="coerce")
df2 = df2.dropna(subset=["PCG_manual"]).copy()

df2["y_3class_str"] = df2["PCG_manual"].astype(int).map(RELABEL_MAP)
df2 = df2.dropna(subset=["y_3class_str"]).copy()

le2 = LabelEncoder()
le2.fit(CLASS_ORDER)
df2["y_3class"] = le2.transform(df2["y_3class_str"].values)

print("\n[Step 2] 3-class distribution:")
print(df2["y_3class_str"].value_counts())

# --- SQI extraction ---


def _safe_float(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def extract_all_pcg_sqi_features(pcg_sig, fs=PCG_FS):
    try:
        x = np.asarray(pcg_sig, dtype=float).squeeze()
        if x.ndim != 1 or len(x) < int(0.5 * fs):
            return {k: np.nan for k in FEATURE_COLS_STEP2}
    except Exception:
        return {k: np.nan for k in FEATURE_COLS_STEP2}

    out = {}
    try:
        out["seSQI"] = _safe_float(
            sqi_pcg_lib.se_sqi_pcg(x, fs, M=SE_M, r=SE_R))
    except Exception:
        out["seSQI"] = np.nan
    try:
        out["cpSQI"] = _safe_float(
            sqi_pcg_lib.correlation_prominence_pcg(x, fs))
    except Exception:
        out["cpSQI"] = np.nan
    try:
        out["pr100_200SQI"] = _safe_float(
            sqi_pcg_lib.pcg_power_ratio_100_200(x, fs))
    except Exception:
        out["pr100_200SQI"] = np.nan
    try:
        out["pr200_400SQI"] = _safe_float(
            sqi_pcg_lib.pcg_power_ratio_200_400(x, fs))
    except Exception:
        out["pr200_400SQI"] = np.nan
    try:
        out["mean_133_267"] = _safe_float(
            sqi_pcg_lib.mfcc_mean_133_267_pcg(x, fs))
    except Exception:
        out["mean_133_267"] = np.nan
    try:
        out["median_133_267"] = _safe_float(
            sqi_pcg_lib.mfcc_median_133_267_pcg(x, fs))
    except Exception:
        out["median_133_267"] = np.nan
    try:
        out["max_600_733"] = _safe_float(
            sqi_pcg_lib.mfcc_max_600_733_pcg(x, fs))
    except Exception:
        out["max_600_733"] = np.nan
    try:
        out["diff_peak_sqi"] = _safe_float(
            sqi_pcg_lib.pcg_periodogram_peak_difference(x, fs))
    except Exception:
        out["diff_peak_sqi"] = np.nan
    try:
        out["svdSQI"] = _safe_float(sqi_pcg_lib.svd_sqi_pcg(
            x, fs, hr_range_bpm=SVD_HR_RANGE_BPM))
    except Exception:
        out["svdSQI"] = np.nan

    return out


print("\n[Step 2] Extracting PCG SQIs...")
feat_df = df2[PCG_SIGNAL_COL].apply(lambda sig: pd.Series(
    extract_all_pcg_sqi_features(sig, fs=PCG_FS)))
df2 = pd.concat([df2.reset_index(drop=True),
                feat_df.reset_index(drop=True)], axis=1)

n_before = len(df2)
df2 = df2.dropna(subset=FEATURE_COLS_STEP2).copy()
print(
    f"[Step 2] Rows before dropping NaNs in SQIs: {n_before}, after: {len(df2)}")

# Save extracted features
features_csv = os.path.join(
    OUT_DIR_STEP2, "step2_pcg_unimodal_features_extracted.csv")
df2[["ID", "Auscultation_Point", "y_3class_str"] +
    FEATURE_COLS_STEP2].to_csv(features_csv, index=False)
print(f"[Step 2] Saved extracted features: {features_csv}")

# --- Train LogReg ---
X2 = df2[FEATURE_COLS_STEP2].values
y2 = df2["y_3class"].values

lr_pipe_step2 = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        multi_class="multinomial",
        solver="lbfgs",
        random_state=RANDOM_STATE
    ))
])

print("\n[Step 2] Train/test evaluation...")
Xtr2, Xte2, ytr2, yte2 = train_test_split(
    X2, y2, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y2)

_OUT_DIR_BACKUP = OUT_DIR
OUT_DIR = OUT_DIR_STEP2

res2_test = evaluate_test_split(
    model=lr_pipe_step2,
    Xtr=Xtr2, Xte=Xte2,
    ytr=ytr2, yte=yte2,
    class_labels=np.array([0, 1, 2], dtype=int),
    class_names=CLASS_ORDER,
    task_name="Step 2 — LogReg (PCG unimodal ALL SQIs) — 3-class",
    out_prefix="step2_logreg_pcg_unimodal_allSQI_3class"
)

print("\n[Step 2] Cross-validated summary...")
res2_cv = cross_validated_summary(
    pipeline=lr_pipe_step2,
    X=X2, y=y2,
    class_labels=np.array([0, 1, 2], dtype=int),
    task_name="Step 2 — LogReg (PCG unimodal ALL SQIs) — 3-class"
)

OUT_DIR = _OUT_DIR_BACKUP

# %% =========================
# %% STEP 3 — ECG unimodal SQIs (ALL) + 3-class Logistic Regression
# %% Signal source: ecg_ulsge.pkl (loaded separately)
# %% Target: manual ECG label from `merged["ECG"]` (Excel column C header 'ECG')
# %% Clean join: temporary normalized keys only (no redundant Trial/Spot/Normalized columns kept)
# %% =========================
OUT_DIR_STEP3 = "exp_step3_ecg_unimodal_allSQI_logreg_3class"
os.makedirs(OUT_DIR_STEP3, exist_ok=True)

ECG_PKL_PATH = r"..\DatasetCHVNGE\ecg_ulsge.pkl"

ECG_FS = 500
ECG_LPF_ORDER = 4
ECG_LPF_FC = 125

# Expected columns in ecg_ulsge.pkl
ECG_ID_COL = "ID"
ECG_POINT_COL = "Auscultation_Point"
ECG_SIGNAL_COL = "ECG"  # signal must be this

# Manual target column name in merged (Excel column C header)
Y_MANUAL_COL = "ECG"                 # as it appears in merged
Y_MANUAL_COL_RENAMED = "ECG_manual"  # avoid any collision (safety)

FEATURE_COLS_STEP3 = [
    "bSQI",
    "pSQI",
    "sSQI",
    "kSQI",
    "fSQI",
    "basSQI",
]

print("\n[Step 3] Configured:")
print(f"  OUT_DIR_STEP3: {OUT_DIR_STEP3}")
print(f"  ECG_PKL_PATH: {ECG_PKL_PATH}")
print(f"  ECG_FS: {ECG_FS}, lowpass: fc={ECG_LPF_FC} (order={ECG_LPF_ORDER})")
print(
    f"  Manual target column: {Y_MANUAL_COL} -> will be renamed to {Y_MANUAL_COL_RENAMED}")

# --- Preconditions ---
if "merged" not in globals():
    raise RuntimeError(
        "Step 3 expects `merged` from Step 1 (clean merged output).")
if Y_MANUAL_COL not in merged.columns:
    raise ValueError(
        f"Manual ECG column '{Y_MANUAL_COL}' not found in merged. Ensure m_quality has 'ECG' column.")
if "pplib" not in globals():
    raise ImportError(
        "preprocessing_lib (pplib) not available. Fix imports to run preprocessing.")

# %% -------------------------
# %% Load ECG signals
# %% -------------------------
print("\n[Step 3] Loading ECG signals...")
ecg_df = pd.read_pickle(ECG_PKL_PATH).copy()
print(f"  ecg_df shape: {ecg_df.shape}")

for col in [ECG_ID_COL, ECG_POINT_COL, ECG_SIGNAL_COL]:
    if col not in ecg_df.columns:
        raise ValueError(f"ecg_df is missing required column '{col}'.")

ecg_df[ECG_ID_COL] = ecg_df[ECG_ID_COL].astype(str)

# %% -------------------------
# %% Build a clean manual-label df (rename label to avoid collisions)
# %% -------------------------
manual_ecg_df = merged[["ID", "Auscultation_Point", Y_MANUAL_COL]].copy()
manual_ecg_df["ID"] = manual_ecg_df["ID"].astype(str)
manual_ecg_df = manual_ecg_df.rename(
    columns={Y_MANUAL_COL: Y_MANUAL_COL_RENAMED})

# Temporary normalized keys for robust join (NOT kept)
ecg_df["_k_point"] = ecg_df[ECG_POINT_COL].astype(
    str).str.replace("_", "", regex=False).str.upper()
manual_ecg_df["_k_point"] = manual_ecg_df["Auscultation_Point"].astype(
    str).str.replace("_", "", regex=False).str.upper()

# %% -------------------------
# %% Join ECG signals with manual ECG labels (clean, collision-safe)
# %% -------------------------
print("[Step 3] Joining signals with manual ECG labels (clean)...")
df3 = pd.merge(
    ecg_df[[ECG_ID_COL, ECG_POINT_COL, "_k_point", ECG_SIGNAL_COL]].copy(),
    manual_ecg_df[["ID", "Auscultation_Point",
                   "_k_point", Y_MANUAL_COL_RENAMED]].copy(),
    left_on=[ECG_ID_COL, "_k_point"],
    right_on=["ID", "_k_point"],
    how="inner",
)

# Clean up: remove join helper + redundant manual ausc point
df3 = df3.drop(columns=["_k_point"])
df3 = df3.drop(columns=["Auscultation_Point_y"]).rename(
    columns={"Auscultation_Point_x": "Auscultation_Point"})

print(f"  joined rows: {len(df3)}")

# Safety: avoid suffix collisions
if "ECG_x" in df3.columns or "ECG_y" in df3.columns:
    raise RuntimeError(
        "Found ECG_x/ECG_y after merge; collision-safe join failed.")

# %% -------------------------
# %% Preprocess ECG (lowpass) BEFORE SQI extraction
# %% -------------------------
print("\n[Step 3] Preprocessing ECG (Butterworth lowpass)...")
df3[ECG_SIGNAL_COL] = df3[ECG_SIGNAL_COL].apply(
    lambda data: pplib.butterworth_filter(
        data,
        filter_topology="lowpass",
        order=ECG_LPF_ORDER,
        fs=ECG_FS,
        fc=ECG_LPF_FC,
    )
)

# %% -------------------------
# %% Build 3-class target from manual annotations (ECG_manual)
# %% -------------------------
df3[Y_MANUAL_COL_RENAMED] = pd.to_numeric(
    df3[Y_MANUAL_COL_RENAMED], errors="coerce")
df3 = df3.dropna(subset=[Y_MANUAL_COL_RENAMED]).copy()

df3["y_3class_str"] = df3[Y_MANUAL_COL_RENAMED].astype(int).map(RELABEL_MAP)
df3 = df3.dropna(subset=["y_3class_str"]).copy()

le3 = LabelEncoder()
le3.fit(CLASS_ORDER)
df3["y_3class"] = le3.transform(df3["y_3class_str"].values)

print("\n[Step 3] 3-class distribution:")
print(df3["y_3class_str"].value_counts())

# %% -------------------------
# %% Extract ALL unimodal ECG SQIs for each signal (ecg_unimodal_sqi_test-style)
# %% -------------------------


def extract_all_ecg_sqi_features(ecg_sig, fs=ECG_FS):
    """
    Extract all unimodal ECG SQIs for one signal.
    Each SQI is isolated in its own try/except so partial results survive.
    """
    try:
        x = np.asarray(ecg_sig, dtype=float).squeeze()
        if x.ndim != 1 or len(x) < int(0.5 * fs):
            return {k: np.nan for k in FEATURE_COLS_STEP3}
    except Exception:
        return {k: np.nan for k in FEATURE_COLS_STEP3}

    out = {}

    # Functions per your snippet
    try:
        out["bSQI"] = _safe_float(sqi_ecg_lib.bSQI(x, fs))
    except Exception:
        out["bSQI"] = np.nan

    try:
        out["pSQI"] = _safe_float(sqi_ecg_lib.pSQI(x, fs))
    except Exception:
        out["pSQI"] = np.nan

    try:
        out["sSQI"] = _safe_float(sqi_ecg_lib.sSQI(x))
    except Exception:
        out["sSQI"] = np.nan

    try:
        out["kSQI"] = _safe_float(sqi_ecg_lib.kSQI(x))
    except Exception:
        out["kSQI"] = np.nan

    try:
        out["fSQI"] = _safe_float(sqi_ecg_lib.fSQI(x, fs))
    except Exception:
        out["fSQI"] = np.nan

    try:
        out["basSQI"] = _safe_float(sqi_ecg_lib.basSQI(x, fs))
    except Exception:
        out["basSQI"] = np.nan

    return out


print("\n[Step 3] Extracting ALL ECG SQIs for all joined signals...")
feat_df3 = df3[ECG_SIGNAL_COL].apply(lambda sig: pd.Series(
    extract_all_ecg_sqi_features(sig, fs=ECG_FS)))
df3 = pd.concat([df3.reset_index(drop=True),
                feat_df3.reset_index(drop=True)], axis=1)

# Drop rows with missing SQIs (strict policy)
n_before = len(df3)
df3 = df3.dropna(subset=FEATURE_COLS_STEP3).copy()
print(
    f"[Step 3] Rows before dropping NaNs in SQIs: {n_before}, after: {len(df3)}")

# Save extracted features (audit/debug)
features3_csv = os.path.join(
    OUT_DIR_STEP3, "step3_ecg_unimodal_features_extracted.csv")
df3[["ID", "Auscultation_Point", "y_3class_str"] +
    FEATURE_COLS_STEP3].to_csv(features3_csv, index=False)
print(f"[Step 3] Saved extracted features: {features3_csv}")

# %% -------------------------
# %% ML: Multinomial Logistic Regression (3-class), using ALL ECG SQIs
# %% -------------------------
X3 = df3[FEATURE_COLS_STEP3].values
y3 = df3["y_3class"].values

lr_pipe_step3 = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        multi_class="multinomial",
        solver="lbfgs",
        random_state=RANDOM_STATE
    ))
])

print("\n[Step 3] Train/test split (stratified) and evaluation...")
Xtr3, Xte3, ytr3, yte3 = train_test_split(
    X3, y3,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y3
)

# evaluate_test_split writes into global OUT_DIR -> temporarily point it to step3 folder
_OUT_DIR_BACKUP = OUT_DIR
OUT_DIR = OUT_DIR_STEP3

res3_test = evaluate_test_split(
    model=lr_pipe_step3,
    Xtr=Xtr3, Xte=Xte3,
    ytr=ytr3, yte=yte3,
    class_labels=np.array([0, 1, 2], dtype=int),
    class_names=CLASS_ORDER,
    task_name="Step 3 — LogReg (ECG unimodal ALL SQIs) — 3-class",
    out_prefix="step3_logreg_ecg_unimodal_allSQI_3class"
)

print("\n[Step 3] Cross-validated summary metrics...")
res3_cv = cross_validated_summary(
    pipeline=lr_pipe_step3,
    X=X3,
    y=y3,
    class_labels=np.array([0, 1, 2], dtype=int),
    task_name="Step 3 — LogReg (ECG unimodal ALL SQIs) — 3-class"
)

# restore OUT_DIR
OUT_DIR = _OUT_DIR_BACKUP

# Save Step 3 summary
summary3_df = pd.DataFrame([
    {
        "experiment": "step3_logreg_ecg_unimodal_allSQI_3class_test",
        "n_samples": int(len(y3)),
        "n_features": int(X3.shape[1]),
        "auc_ovr": res3_test["auc_ovr"],
        "accuracy": res3_test["accuracy"],
        "macro_sensitivity": res3_test["macro_sensitivity"],
        "macro_specificity": res3_test["macro_specificity"],
        "macro_f1": res3_test["macro_f1"],
    },
    {
        "experiment": "step3_logreg_ecg_unimodal_allSQI_3class_cv",
        "n_samples": int(len(y3)),
        "n_features": int(X3.shape[1]),
        "auc_ovr": res3_cv["cv_auc_mean"],
        "accuracy": res3_cv["cv_acc_mean"],
        "macro_sensitivity": res3_cv["cv_rec_mean"],
        "macro_specificity": res3_cv["cv_spec_mean"],
        "macro_f1": res3_cv["cv_f1_mean"],
    }
])

summary3_csv = os.path.join(
    OUT_DIR_STEP3, "step3_logreg_ecg_unimodal_allSQI_3class_summary.csv")
summary3_df.to_csv(summary3_csv, index=False)

print("\n" + "=" * 80)
print("[Step 3 COMPLETE] SUMMARY TABLE")
print("=" * 80)
print(summary3_df.to_string(index=False))
print(f"\nSaved: {summary3_csv}")
print(f"Artifacts folder: {OUT_DIR_STEP3}")

# %% =========================
# %% STEP 4 — Multimodal fusion (MIN) of ECG + PCG predictions
# %% NO refitting — inference only
# %% Compare vs Step 1 LogReg
# %% + Per-class metrics section (requested)
# %% =========================

OUT_DIR_STEP4 = "exp_step4_fusion_min_vs_step1"
os.makedirs(OUT_DIR_STEP4, exist_ok=True)

print("\n[Step 4] Multimodal fusion: MIN(PCG_pred, ECG_pred) vs mSQA_min")

# ------------------------------------------------------------------
# Align samples: use SAME TEST SET as Step 1
# ------------------------------------------------------------------


def make_key(df):
    return df["ID"].astype(str) + "||" + df["Auscultation_Point"].astype(str)


# Keep keys consistent
df1["_key"] = make_key(df1)
df2["_key"] = make_key(df2)
df3["_key"] = make_key(df3)

# Rebuild the Step-1 test keys by recomputing the split on df1 (same seed & stratification),
# then using the resulting indices. This avoids relying on yte.index existence.
alignment_metrics = [c for c in df1.columns if str(
    c).startswith("alignment_metric")]
if not alignment_metrics:
    raise ValueError("[Step 4] alignment_metric* columns not found in df1.")

X1_all = df1[alignment_metrics].values
y1_all = df1["y_3class"].values.astype(int)
idx_all = np.arange(len(df1))

_, idx_te = train_test_split(
    idx_all,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y1_all
)

test_keys = set(df1.iloc[idx_te]["_key"].values)

# Subset test rows for each df and strictly align by key
df1_test = df1[df1["_key"].isin(test_keys)].copy().sort_values("_key")
df2_test = df2[df2["_key"].isin(test_keys)].copy().sort_values("_key")
df3_test = df3[df3["_key"].isin(test_keys)].copy().sort_values("_key")

# Assert alignment
if not (df1_test["_key"].values == df2_test["_key"].values).all() or not (df1_test["_key"].values == df3_test["_key"].values).all():
    raise RuntimeError(
        "[Step 4] Key misalignment between df1_test/df2_test/df3_test. "
        "This means the test subsets do not perfectly overlap."
    )

y_true = df1_test["y_3class"].values.astype(int)

# ------------------------------------------------------------------
# Inference ONLY (no fitting)
# ------------------------------------------------------------------
X2_test = df2_test[FEATURE_COLS_STEP2].values
X3_test = df3_test[FEATURE_COLS_STEP3].values

yhat_pcg = lr_pipe_step2.predict(X2_test).astype(int)
yhat_ecg = lr_pipe_step3.predict(X3_test).astype(int)

# Fusion rule (ordinal MIN)
yhat_fused = np.minimum(yhat_pcg, yhat_ecg).astype(int)

# ------------------------------------------------------------------
# Metrics (aggregated)
# ------------------------------------------------------------------
labels_int = np.array([0, 1, 2], dtype=int)

cm_fused = confusion_matrix(y_true, yhat_fused, labels=labels_int)

acc_fused = accuracy_score(y_true, yhat_fused)
f1_fused = f1_score(y_true, yhat_fused, average="macro")
sens_fused = recall_score(y_true, yhat_fused, average="macro")
spec_fused = float(np.mean(compute_specificity_from_cm(cm_fused)))

# AUC OvR
y_true_bin = label_binarize(y_true, classes=labels_int)
yhat_bin = label_binarize(yhat_fused, classes=labels_int)
auc_fused = roc_auc_score(y_true_bin, yhat_bin, multi_class="ovr")

print("\n" + "=" * 70)
print("STEP 4 — FUSED MIN(PCG, ECG) vs mSQA_min (TEST SET)")
print("=" * 70)
print(f"AUC OvR:            {auc_fused:.4f}")
print(f"Accuracy:           {acc_fused:.4f}")
print(f"Macro Sensitivity:  {sens_fused:.4f}")
print(f"Macro Specificity:  {spec_fused:.4f}")
print(f"Macro F1:           {f1_fused:.4f}")

# ------------------------------------------------------------------
# Per-class metrics
# ------------------------------------------------------------------
per_cls_fused = per_class_metrics_from_cm(cm_fused, CLASS_ORDER)

print("\n" + "-" * 70)
print("STEP 4 — PER-CLASS METRICS (Fusion MIN)")
print("-" * 70)
print(per_cls_fused.to_string(index=False))

per_cls_csv = os.path.join(
    OUT_DIR_STEP4, "step4_fusion_min_per_class_metrics.csv")
per_cls_fused.to_csv(per_cls_csv, index=False)
print(f"\nSaved per-class metrics: {per_cls_csv}")

# Confusion matrix (%)
cm_png = os.path.join(OUT_DIR_STEP4, "step4_fusion_min_cm_percent.png")
save_confusion_matrix_percent(
    y_true=y_true,
    y_pred=yhat_fused,
    labels=labels_int,
    display_labels=CLASS_ORDER,
    title="Fusion MIN(PCG, ECG) vs mSQA_min — Confusion Matrix (%)",
    out_png=cm_png
)

# ------------------------------------------------------------------
# Comparison vs STEP 1 metrics (already computed in res1_test)
# ------------------------------------------------------------------
comparison_df = pd.DataFrame([
    {
        "approach": "Step 1 — Alignment LogReg",
        "accuracy": res1_test["accuracy"],
        "macro_f1": res1_test["macro_f1"],
        "macro_sensitivity": res1_test["macro_sensitivity"],
        "macro_specificity": res1_test["macro_specificity"],
        "auc_ovr": res1_test["auc_ovr"],
    },
    {
        "approach": "Step 4 — Fusion MIN(PCG, ECG)",
        "accuracy": acc_fused,
        "macro_f1": f1_fused,
        "macro_sensitivity": sens_fused,
        "macro_specificity": spec_fused,
        "auc_ovr": auc_fused,
    },
])

comparison_csv = os.path.join(OUT_DIR_STEP4, "step4_vs_step1_comparison.csv")
comparison_df.to_csv(comparison_csv, index=False)

print("\n" + "=" * 70)
print("STEP 4 vs STEP 1 — METRIC COMPARISON")
print("=" * 70)
print(comparison_df.to_string(index=False))
print(f"\nSaved comparison table: {comparison_csv}")
print(f"Artifacts folder: {OUT_DIR_STEP4}")


# %% =========================
# %% STEP 5 — Feature-level multimodal fusion (PCG SQIs + ECG SQIs)
# %% Ground truth: mSQA_min
# %% Model: Multinomial Logistic Regression
# %% =========================

OUT_DIR_STEP5 = "exp_step5_multimodal_features_logreg_3class"
os.makedirs(OUT_DIR_STEP5, exist_ok=True)

print("\n[Step 5] Multimodal feature fusion (PCG SQIs + ECG SQIs) vs mSQA_min")

# ------------------------------------------------------------------
# Build consistent keys
# ------------------------------------------------------------------


def make_key(df):
    return df["ID"].astype(str) + "||" + df["Auscultation_Point"].astype(str)


df1["_key"] = make_key(df1)   # Step 1 (mSQA_min ground truth)
df2["_key"] = make_key(df2)   # PCG SQIs
df3["_key"] = make_key(df3)   # ECG SQIs

# ------------------------------------------------------------------
# Merge ALL features with mSQA_min ground truth
# ------------------------------------------------------------------
print("[Step 5] Merging feature tables...")

df5 = (
    df1[["_key", "ID", "Auscultation_Point", "y_3class", "y_3class_str"]]
    .merge(
        df2[["_key"] + FEATURE_COLS_STEP2],
        on="_key",
        how="inner",
    )
    .merge(
        df3[["_key"] + FEATURE_COLS_STEP3],
        on="_key",
        how="inner",
    )
)

print(f"  rows after merge: {len(df5)}")
print(
    f"  total features:  {len(FEATURE_COLS_STEP2) + len(FEATURE_COLS_STEP3)}")

# ------------------------------------------------------------------
# Feature matrix and target
# ------------------------------------------------------------------
FEATURE_COLS_STEP5 = FEATURE_COLS_STEP2 + FEATURE_COLS_STEP3

X5 = df5[FEATURE_COLS_STEP5].values
y5 = df5["y_3class"].values.astype(int)

print("\n[Step 5] 3-class distribution:")
print(df5["y_3class_str"].value_counts())

# ------------------------------------------------------------------
# Model (same configuration as previous steps)
# ------------------------------------------------------------------
lr_pipe_step5 = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        multi_class="multinomial",
        solver="lbfgs",
        random_state=RANDOM_STATE
    ))
])

# ------------------------------------------------------------------
# Train / Test split (stratified)
# ------------------------------------------------------------------
print("\n[Step 5] Train/test evaluation...")
Xtr5, Xte5, ytr5, yte5 = train_test_split(
    X5, y5,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y5
)

# Redirect outputs
_OUT_DIR_BACKUP = OUT_DIR
OUT_DIR = OUT_DIR_STEP5

res5_test = evaluate_test_split(
    model=lr_pipe_step5,
    Xtr=Xtr5, Xte=Xte5,
    ytr=ytr5, yte=yte5,
    class_labels=np.array([0, 1, 2], dtype=int),
    class_names=CLASS_ORDER,
    task_name="Step 5 — LogReg (PCG SQIs + ECG SQIs) → mSQA_min",
    out_prefix="step5_multimodal_features_logreg_3class"
)

# ------------------------------------------------------------------
# Cross-validated summary
# ------------------------------------------------------------------
print("\n[Step 5] Cross-validated summary metrics...")
res5_cv = cross_validated_summary(
    pipeline=lr_pipe_step5,
    X=X5,
    y=y5,
    class_labels=np.array([0, 1, 2], dtype=int),
    task_name="Step 5 — LogReg (PCG SQIs + ECG SQIs) → mSQA_min"
)

# Restore OUT_DIR
OUT_DIR = _OUT_DIR_BACKUP

# ------------------------------------------------------------------
# Save summary table
# ------------------------------------------------------------------
summary5_df = pd.DataFrame([
    {
        "experiment": "step5_multimodal_features_logreg_test",
        "n_samples": int(len(y5)),
        "n_features": int(X5.shape[1]),
        "auc_ovr": res5_test["auc_ovr"],
        "accuracy": res5_test["accuracy"],
        "macro_sensitivity": res5_test["macro_sensitivity"],
        "macro_specificity": res5_test["macro_specificity"],
        "macro_f1": res5_test["macro_f1"],
    },
    {
        "experiment": "step5_multimodal_features_logreg_cv",
        "n_samples": int(len(y5)),
        "n_features": int(X5.shape[1]),
        "auc_ovr": res5_cv["cv_auc_mean"],
        "accuracy": res5_cv["cv_acc_mean"],
        "macro_sensitivity": res5_cv["cv_rec_mean"],
        "macro_specificity": res5_cv["cv_spec_mean"],
        "macro_f1": res5_cv["cv_f1_mean"],
    }
])

summary5_csv = os.path.join(
    OUT_DIR_STEP5,
    "step5_multimodal_features_logreg_3class_summary.csv"
)
summary5_df.to_csv(summary5_csv, index=False)

print("\n" + "=" * 80)
print("[Step 5 COMPLETE] SUMMARY TABLE")
print("=" * 80)
print(summary5_df.to_string(index=False))
print(f"\nSaved: {summary5_csv}")
print(f"Artifacts folder: {OUT_DIR_STEP5}")
# %% =========================
# %% STEP 6 — Unified comparison (macro + per-class) for Steps 1, 2, 3, 4, 5
# %% + Statistical significance tests (paired t-tests, no Bonferroni) for Steps 1 vs 4 and 1 vs 5
# %% Notes:
# %%   - No deltas.
# %%   - t-tests are computed on PER-SAMPLE metrics (paired), not on single aggregate scalars.
# %%     This yields a valid sample size for the test without refitting or CV loops.
# %% =========================

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

OUT_DIR_STEP6 = "exp_step6_unified_comparison"
os.makedirs(OUT_DIR_STEP6, exist_ok=True)

print("\n[Step 6] Building unified comparison tables (macro + per-class) for Steps 1–5...")
print("[Step 6] Running paired t-tests (no Bonferroni) for Steps 1 vs 4 and Steps 1 vs 5...")

# ------------------------------------------------------------------
# Preconditions (expected objects from previous steps)
# ------------------------------------------------------------------
# Step 1
if "res1_test" not in globals() or "per_class_df" not in res1_test:
    raise RuntimeError("[Step 6] Missing res1_test['per_class_df']. Ensure Step 1 ran fully.")

# Step 2
if "res2_test" not in globals() or "per_class_df" not in res2_test:
    raise RuntimeError("[Step 6] Missing res2_test['per_class_df']. Ensure Step 2 ran fully.")

# Step 3
if "res3_test" not in globals() or "per_class_df" not in res3_test:
    raise RuntimeError("[Step 6] Missing res3_test['per_class_df']. Ensure Step 3 ran fully.")

# Step 4
if "per_cls_fused" not in globals():
    raise RuntimeError("[Step 6] Missing per_cls_fused. Ensure Step 4 ran with per-class metrics section.")
if not all(v in globals() for v in ["auc_fused", "acc_fused", "sens_fused", "spec_fused", "f1_fused"]):
    raise RuntimeError("[Step 6] Missing Step 4 macro metrics (auc_fused, acc_fused, sens_fused, spec_fused, f1_fused).")

# Step 5
if "res5_test" not in globals() or "per_class_df" not in res5_test:
    raise RuntimeError("[Step 6] Missing res5_test['per_class_df']. Ensure Step 5 ran fully.")

# For t-tests we need per-sample predictions/labels from Step 1, 4, 5 on the SAME aligned test set
# Step 4 defines: y_true, yhat_fused
# Step 1 must expose its test-set truth/preds; we will recompute Step 1 preds on df1_test to avoid changing Step 1 code.
if not all(v in globals() for v in ["df1_test", "df2_test", "df3_test", "y_true", "yhat_fused"]):
    raise RuntimeError(
        "[Step 6] Missing Step 4 aligned test-set variables (df1_test/df2_test/df3_test/y_true/yhat_fused). "
        "Run Step 4 first."
    )
if "lr_pipe" not in globals():
    raise RuntimeError("[Step 6] Missing lr_pipe (Step 1 model pipeline). Step 1 must have defined it.")
if "lr_pipe_step5" not in globals():
    raise RuntimeError("[Step 6] Missing lr_pipe_step5 (Step 5 model pipeline). Step 5 must have defined it.")
if "FEATURE_COLS_STEP2" not in globals() or "FEATURE_COLS_STEP3" not in globals():
    raise RuntimeError("[Step 6] Missing FEATURE_COLS_STEP2/FEATURE_COLS_STEP3. Ensure Steps 2 and 3 ran.")

# ------------------------------------------------------------------
# 1) MACRO metrics table (Steps 1–5) — NO deltas
# ------------------------------------------------------------------
macro_df = pd.DataFrame([
    {
        "approach": "Step 1 — Alignment metrics (LogReg)",
        "fusion_level": "Physiological / temporal",
        "auc_ovr": res1_test["auc_ovr"],
        "accuracy": res1_test["accuracy"],
        "macro_sensitivity": res1_test["macro_sensitivity"],
        "macro_specificity": res1_test["macro_specificity"],
        "macro_f1": res1_test["macro_f1"],
    },
    {
        "approach": "Step 2 — PCG unimodal SQIs (LogReg)",
        "fusion_level": "Unimodal (PCG)",
        "auc_ovr": res2_test["auc_ovr"],
        "accuracy": res2_test["accuracy"],
        "macro_sensitivity": res2_test["macro_sensitivity"],
        "macro_specificity": res2_test["macro_specificity"],
        "macro_f1": res2_test["macro_f1"],
    },
    {
        "approach": "Step 3 — ECG unimodal SQIs (LogReg)",
        "fusion_level": "Unimodal (ECG)",
        "auc_ovr": res3_test["auc_ovr"],
        "accuracy": res3_test["accuracy"],
        "macro_sensitivity": res3_test["macro_sensitivity"],
        "macro_specificity": res3_test["macro_specificity"],
        "macro_f1": res3_test["macro_f1"],
    },
    {
        "approach": "Step 4 — Decision fusion MIN(ECG, PCG)",
        "fusion_level": "Decision-level",
        "auc_ovr": auc_fused,
        "accuracy": acc_fused,
        "macro_sensitivity": sens_fused,
        "macro_specificity": spec_fused,
        "macro_f1": f1_fused,
    },
    {
        "approach": "Step 5 — Feature fusion (ECG SQIs + PCG SQIs)",
        "fusion_level": "Feature-level",
        "auc_ovr": res5_test["auc_ovr"],
        "accuracy": res5_test["accuracy"],
        "macro_sensitivity": res5_test["macro_sensitivity"],
        "macro_specificity": res5_test["macro_specificity"],
        "macro_f1": res5_test["macro_f1"],
    },
])

macro_csv = os.path.join(OUT_DIR_STEP6, "macro_comparison_steps_1_to_5.csv")
macro_df.to_csv(macro_csv, index=False)

print("\n" + "=" * 120)
print("MACRO COMPARISON (Steps 1–5)")
print("=" * 120)
print(macro_df.to_string(index=False))
print(f"\nSaved: {macro_csv}")

# ------------------------------------------------------------------
# 2) PER-CLASS comparison tables (Steps 1–5) — NO deltas
# ------------------------------------------------------------------
def _prep_per_class(df_per: pd.DataFrame, approach: str) -> pd.DataFrame:
    dfp = df_per.copy()
    expected = ["class", "support", "precision", "sensitivity_recall", "specificity", "f1"]
    missing = [c for c in expected if c not in dfp.columns]
    if missing:
        raise RuntimeError(
            f"[Step 6] Per-class df missing columns {missing} for approach={approach}. "
            f"Columns: {list(dfp.columns)}"
        )
    dfp["approach"] = approach
    return dfp[["approach"] + expected]

pc_step1 = _prep_per_class(res1_test["per_class_df"], "Step 1 — Alignment metrics (LogReg)")
pc_step2 = _prep_per_class(res2_test["per_class_df"], "Step 2 — PCG unimodal SQIs (LogReg)")
pc_step3 = _prep_per_class(res3_test["per_class_df"], "Step 3 — ECG unimodal SQIs (LogReg)")
pc_step4 = _prep_per_class(per_cls_fused,              "Step 4 — Decision fusion MIN(ECG, PCG)")
pc_step5 = _prep_per_class(res5_test["per_class_df"], "Step 5 — Feature fusion (ECG SQIs + PCG SQIs)")

per_class_long = pd.concat([pc_step1, pc_step2, pc_step3, pc_step4, pc_step5], axis=0, ignore_index=True)

wide_cols = ["precision", "sensitivity_recall", "specificity", "f1"]
per_class_wide = per_class_long.pivot_table(
    index="class",
    columns="approach",
    values=wide_cols,
    aggfunc="first"
)
per_class_wide.columns = [f"{m}__{a}" for (m, a) in per_class_wide.columns]
per_class_wide = per_class_wide.reset_index()

per_class_long_csv = os.path.join(OUT_DIR_STEP6, "per_class_comparison_long_steps_1_to_5.csv")
per_class_wide_csv = os.path.join(OUT_DIR_STEP6, "per_class_comparison_wide_steps_1_to_5.csv")

per_class_long.to_csv(per_class_long_csv, index=False)
per_class_wide.to_csv(per_class_wide_csv, index=False)

print("\n" + "=" * 120)
print("PER-CLASS COMPARISON (LONG) — Steps 1–5")
print("=" * 120)
print(per_class_long.to_string(index=False))
print(f"\nSaved (long): {per_class_long_csv}")

print("\n" + "=" * 120)
print("PER-CLASS COMPARISON (WIDE) — Steps 1–5")
print("=" * 120)
print(per_class_wide.to_string(index=False))
print(f"\nSaved (wide): {per_class_wide_csv}")

# ------------------------------------------------------------------
# 3) Statistical significance tests (paired t-tests, no Bonferroni)
#    for MACRO metrics of Steps 1 vs 4, and Steps 1 vs 5
#    and PER-CLASS metrics of Steps 1 vs 4, and Steps 1 vs 5
# ------------------------------------------------------------------
print("\n" + "=" * 120)
print("STATISTICAL SIGNIFICANCE TESTS (paired t-test, no Bonferroni)")
print("=" * 120)

# ---- Build aligned predictions for Step 1 and Step 5 on the SAME Step-4 test set ----
# Step 4 already aligned the test set as df1_test and defined y_true
labels_int = np.array([0, 1, 2], dtype=int)

# Step 1 predictions on df1_test
alignment_metrics = [c for c in df1_test.columns if str(c).startswith("alignment_metric")]
if not alignment_metrics:
    raise RuntimeError("[Step 6] alignment_metric* not found in df1_test for Step 1 inference.")
X1_test = df1_test[alignment_metrics].values
yhat_step1 = lr_pipe.predict(X1_test).astype(int)

# Step 5 predictions on df1_test (needs combined features, aligned via df2_test/df3_test)
X2_test = df2_test[FEATURE_COLS_STEP2].values
X3_test = df3_test[FEATURE_COLS_STEP3].values
X5_test = np.concatenate([X2_test, X3_test], axis=1)
yhat_step5 = lr_pipe_step5.predict(X5_test).astype(int)

# Step 4 already has yhat_fused
yhat_step4 = yhat_fused.astype(int)

y_true_aligned = y_true.astype(int)

# ---- Per-sample "macro" vectors for paired t-tests (valid sample size without CV/refitting) ----
def per_sample_accuracy(y_t, y_p):
    return (y_t == y_p).astype(float)

def per_sample_f1_ovr(y_t, y_p, cls):
    # One-vs-rest F1 computed per sample is not well-defined (needs aggregation).
    # We therefore use correctness vectors for macro tests and do class-wise t-tests below.
    raise NotImplementedError

acc_vec_1 = per_sample_accuracy(y_true_aligned, yhat_step1)
acc_vec_4 = per_sample_accuracy(y_true_aligned, yhat_step4)
acc_vec_5 = per_sample_accuracy(y_true_aligned, yhat_step5)

# Paired t-tests for sample-wise accuracy (macro proxy)
t14, p14 = ttest_rel(acc_vec_1, acc_vec_4)
t15, p15 = ttest_rel(acc_vec_1, acc_vec_5)

macro_ttest_df = pd.DataFrame([
    {"comparison": "Step 4 vs Step 1", "metric": "sample-wise accuracy", "t_stat": t14, "p_value": p14, "n": len(acc_vec_1)},
    {"comparison": "Step 5 vs Step 1", "metric": "sample-wise accuracy", "t_stat": t15, "p_value": p15, "n": len(acc_vec_1)},
])

macro_ttest_csv = os.path.join(OUT_DIR_STEP6, "ttest_macro_vs_step1.csv")
macro_ttest_df.to_csv(macro_ttest_csv, index=False)

print("\nPaired t-tests — MACRO (sample-wise accuracy)")
print(macro_ttest_df.to_string(index=False))
print(f"\nSaved: {macro_ttest_csv}")

# ---- PER-CLASS paired t-tests (one-vs-rest correctness per class) ----
# For each class, define a paired correctness vector: 1 if predicted class == true class == c, else 0
def per_sample_correct_for_class(y_t, y_p, c):
    return ((y_t == c) & (y_p == c)).astype(float)

rows = []
for c, cname in enumerate(CLASS_ORDER):
    v1 = per_sample_correct_for_class(y_true_aligned, yhat_step1, c)
    v4 = per_sample_correct_for_class(y_true_aligned, yhat_step4, c)
    v5 = per_sample_correct_for_class(y_true_aligned, yhat_step5, c)

    t_c14, p_c14 = ttest_rel(v1, v4)
    t_c15, p_c15 = ttest_rel(v1, v5)

    rows.append({"class": cname, "comparison": "Step 4 vs Step 1", "metric": "class-correctness", "t_stat": t_c14, "p_value": p_c14, "n": len(v1)})
    rows.append({"class": cname, "comparison": "Step 5 vs Step 1", "metric": "class-correctness", "t_stat": t_c15, "p_value": p_c15, "n": len(v1)})

per_class_ttest_df = pd.DataFrame(rows)

per_class_ttest_csv = os.path.join(OUT_DIR_STEP6, "ttest_per_class_vs_step1.csv")
per_class_ttest_df.to_csv(per_class_ttest_csv, index=False)

print("\nPaired t-tests — PER-CLASS (one-vs-rest class-correctness per sample)")
print(per_class_ttest_df.to_string(index=False))
print(f"\nSaved: {per_class_ttest_csv}")

print(f"\nArtifacts folder: {OUT_DIR_STEP6}")
