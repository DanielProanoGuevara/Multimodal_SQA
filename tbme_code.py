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

# %% Global config (edit as needed)
AQ_PATH = r"..\ulsge_quality_metrics.pkl"
MQ_PATH = r"..\ulsge_manual_sqa.xlsx"

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
CLASS_ORDER = ["low_quality", "uncertain", "high_quality"]  # fixed display order for reports

print("Configured:")
print(f"  AQ_PATH: {AQ_PATH}")
print(f"  MQ_PATH: {MQ_PATH}")
print(f"  OUT_DIR: {OUT_DIR}")
print(f"  TEST_SIZE: {TEST_SIZE}, N_SPLITS_CV: {N_SPLITS_CV}")
print(f"  CLASS_ORDER: {CLASS_ORDER}")

# %% =========================
# %% Utilities (reused by all steps)
# %% =========================

def merge_quality_dataframes(ex1_quality: pd.DataFrame, m_quality: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ex1_quality and m_quality using the same key logic as your codebase:
      - ex1_quality: (ID, Normalized_Point) where Normalized_Point is ausc point without '_' and upper
      - m_quality:   (Trial, Normalized_Spot) where Normalized_Spot is spot without '_' and upper
    """
    ex1 = ex1_quality.copy()
    m = m_quality.copy()

    # --- Key normalization ---
    ex1["ID"] = ex1["ID"].astype(str)
    if "Trial" not in m.columns:
        raise ValueError("m_quality must contain a 'Trial' column.")
    m["Trial"] = m["Trial"].astype(str)

    if "Auscultation_Point" not in ex1.columns:
        raise ValueError("ex1_quality must contain 'Auscultation_Point'.")
    if "Spot" not in m.columns:
        raise ValueError("m_quality must contain 'Spot' column.")

    ex1["Normalized_Point"] = ex1["Auscultation_Point"].astype(str).str.replace("_", "", regex=False).str.upper()
    m["Normalized_Spot"] = m["Spot"].astype(str).str.replace("_", "", regex=False).str.upper()

    merged = pd.merge(
        ex1,
        m,
        left_on=["ID", "Normalized_Point"],
        right_on=["Trial", "Normalized_Spot"],
        how="inner",
    )
    return merged


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
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)

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
        cm_percent = np.where(row_sums > 0, cm.astype(float) / row_sums * 100.0, 0.0)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=display_labels)
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
        raise AttributeError("Model/pipeline must implement predict_proba to compute AUC OvR.")
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
    skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

    aucs, accs, recs, f1s, specs = [], [], [], [], []

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        pipeline.fit(Xtr, ytr)
        yhat = pipeline.predict(Xte)

        if not hasattr(pipeline, "predict_proba"):
            raise AttributeError("Pipeline must implement predict_proba to compute AUC OvR.")
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

        print(f"  CV fold {fold}/{N_SPLITS_CV}: AUC={auc:.4f}, Acc={acc:.4f}, Rec={rec:.4f}, Spec={spec:.4f}, F1={f1m:.4f}")

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
            raise ValueError("m_quality must have at least 5 columns to access Excel column E.")
        colE_name = m_quality_cols[4]  # Excel column E -> index 4
        if colE_name not in merged_df.columns:
            raise ValueError(f"Column E name '{colE_name}' not found after merge.")
        y_raw_col = colE_name
        print(f"\nRaw target: using m_quality column E -> '{y_raw_col}' (by index).")

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
# %% STEP 1 — Quantized-only (3-class) Logistic Regression on alignment metrics
# %% =========================
print("\n[Step 1] Loading datasets...")
ex1_quality = pd.read_pickle(AQ_PATH)
m_quality = pd.read_excel(MQ_PATH)

print(f"  ex1_quality shape: {ex1_quality.shape}")
print(f"  m_quality shape:   {m_quality.shape}")

print("\n[Step 1] Merging...")
merged = merge_quality_dataframes(ex1_quality, m_quality)
print(f"  merged rows:    {len(merged)}")
print(f"  merged columns: {len(merged.columns)}")

# Collect alignment metrics automatically (your convention)
alignment_metrics = [c for c in merged.columns if str(c).startswith("alignment_metric")]
if not alignment_metrics:
    raise ValueError("No alignment_metric* columns found in merged DataFrame.")

print(f"\n[Step 1] Found {len(alignment_metrics)} alignment metrics.")
for c in alignment_metrics:
    print(f"  - {c}")

print("\n[Step 1] Building 3-class target...")
df1, y_raw_col = build_3class_target_from_mquality(merged, m_quality.columns)

# Drop rows missing any features (critical before training)
df1 = df1.dropna(subset=alignment_metrics + ["y_3class"]).copy()

print(f"  using raw y col: {y_raw_col}")
print(f"  rows after dropna(features+y): {len(df1)}")
print("  3-class distribution:")
print(df1["y_3class_str"].value_counts())

X1 = df1[alignment_metrics].values
y1 = df1["y_3class"].values

class_names = CLASS_ORDER
class_labels_int = np.array([0, 1, 2], dtype=int)

# Multinomial LR (balanced) with feature scaling
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

print("\n[Step 1] Train/test split (stratified) and evaluation...")
Xtr, Xte, ytr, yte = train_test_split(
    X1, y1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y1
)

res1_test = evaluate_test_split(
    model=lr_pipe,
    Xtr=Xtr, Xte=Xte,
    ytr=ytr, yte=yte,
    class_labels=class_labels_int,
    class_names=class_names,
    task_name="Step 1 — LogReg (alignment metrics) — 3-class",
    out_prefix="step1_logreg_3class"
)

print("\n[Step 1] Cross-validated summary metrics...")
res1_cv = cross_validated_summary(
    pipeline=lr_pipe,
    X=X1,
    y=y1,
    class_labels=class_labels_int,
    task_name="Step 1 — LogReg (alignment metrics) — 3-class"
)

summary1_df = pd.DataFrame([
    {
        "experiment": "step1_logreg_3class_test",
        "auc_ovr": res1_test["auc_ovr"],
        "accuracy": res1_test["accuracy"],
        "macro_sensitivity": res1_test["macro_sensitivity"],
        "macro_specificity": res1_test["macro_specificity"],
        "macro_f1": res1_test["macro_f1"],
        "n_samples": int(len(y1)),
        "n_features": int(X1.shape[1]),
    },
    {
        "experiment": "step1_logreg_3class_cv",
        "auc_ovr": res1_cv["cv_auc_mean"],
        "accuracy": res1_cv["cv_acc_mean"],
        "macro_sensitivity": res1_cv["cv_rec_mean"],
        "macro_specificity": res1_cv["cv_spec_mean"],
        "macro_f1": res1_cv["cv_f1_mean"],
        "n_samples": int(len(y1)),
        "n_features": int(X1.shape[1]),
    }
])

summary1_csv = os.path.join(OUT_DIR, "step1_logreg_3class_summary.csv")
summary1_df.to_csv(summary1_csv, index=False)

print("\n" + "=" * 80)
print("[Step 1 COMPLETE] SUMMARY TABLE")
print("=" * 80)
print(summary1_df.to_string(index=False))
print(f"\nSaved: {summary1_csv}")
print(f"Artifacts folder: {OUT_DIR}")

# %% =========================
# %% STEP 2 — PCG unimodal SQIs (ALL) + 3-class Logistic Regression
# %% Signal source: pcg_ulsge.pkl (loaded separately)
# %% Target: manual PCG label from Excel column D header 'PCG'
# %% IMPORTANT: avoid PCG_x/PCG_y collision by renaming manual label -> PCG_manual
# %% =========================

# --- Optional imports (warn if missing as requested) ---
try:
    import sqi_pcg_lib
except Exception as e:
    warnings.warn(f"[WARN] Could not import sqi_pcg_lib: {e}. Step 2 will fail until fixed.")

try:
    import preprocessing_lib as pplib
except Exception as e:
    warnings.warn(f"[WARN] Could not import preprocessing_lib as pplib: {e}. Step 2 will fail until fixed.")

# --- Step 2 config ---
OUT_DIR_STEP2 = "exp_step2_pcg_unimodal_allSQI_logreg_3class"
os.makedirs(OUT_DIR_STEP2, exist_ok=True)

PCG_PKL_PATH = r"..\DatasetCHVNGE\pcg_ulsge.pkl"

PCG_FS = 3000
PCG_BPF_ORDER = 4
PCG_BPF_FC = [50, 250]

# SQI parameters aligned with your pcg_unimodal_sqi_test usage
SE_M = 2
SE_R = 0.0008
SVD_HR_RANGE_BPM = (70, 220)

# Expected columns in pcg_ulsge.pkl
PCG_ID_COL = "ID"
PCG_POINT_COL = "Auscultation_Point"
PCG_SIGNAL_COL = "PCG"  # must remain the signal column name

# Manual target column name (Excel column D header) -> rename to avoid collision
Y_MANUAL_COL = "PCG"               # as it appears in merged
Y_MANUAL_COL_RENAMED = "PCG_manual"

FEATURE_COLS_STEP2 = [
    "seSQI",
    "cpSQI",
    "pr100_200SQI",
    "pr200_400SQI",
    "mean_133_267",
    "median_133_267",
    "max_600_733",
    "diff_peak_sqi",
    "svdSQI",
]

print("\n[Step 2] Configured:")
print(f"  OUT_DIR_STEP2: {OUT_DIR_STEP2}")
print(f"  PCG_PKL_PATH: {PCG_PKL_PATH}")
print(f"  PCG_FS: {PCG_FS}, bandpass: {PCG_BPF_FC} (order={PCG_BPF_ORDER})")
print(f"  Manual target column: {Y_MANUAL_COL} -> will be renamed to {Y_MANUAL_COL_RENAMED}")

# --- Preconditions: `merged` must exist from Step 1 ---
if "merged_df" in globals() and "merged" not in globals():
    merged = merged_df
if "merged" not in globals():
    raise RuntimeError("Step 2 expects a merged dataframe named `merged` from Step 1.")

if Y_MANUAL_COL not in merged.columns:
    raise ValueError(
        f"Manual target column '{Y_MANUAL_COL}' not found in merged dataframe. "
        "Check the Excel header and merge."
    )

# %% -------------------------
# %% Load PCG dataframe (signals)
# %% -------------------------
print("\n[Step 2] Loading PCG dataframe...")
pcg_df = pd.read_pickle(PCG_PKL_PATH)
print(f"  pcg_df shape: {pcg_df.shape}")

for col in [PCG_ID_COL, PCG_POINT_COL, PCG_SIGNAL_COL]:
    if col not in pcg_df.columns:
        raise ValueError(f"pcg_df is missing required column '{col}'.")

# Normalize join keys (consistent with Step 1)
pcg_df = pcg_df.copy()
pcg_df[PCG_ID_COL] = pcg_df[PCG_ID_COL].astype(str)
pcg_df["Normalized_Point"] = pcg_df[PCG_POINT_COL].astype(str).str.replace("_", "", regex=False).str.upper()

merged_step2 = merged.copy()
merged_step2["ID"] = merged_step2["ID"].astype(str)
merged_step2["Normalized_Point"] = merged_step2["Auscultation_Point"].astype(str).str.replace("_", "", regex=False).str.upper()

# Keep only manual fields needed and RENAME label column to avoid collision with signal 'PCG'
manual_df = merged_step2[["ID", "Auscultation_Point", "Normalized_Point", Y_MANUAL_COL]].copy()
manual_df = manual_df.rename(columns={Y_MANUAL_COL: Y_MANUAL_COL_RENAMED})

# %% -------------------------
# %% Join signals with manual labels (collision-safe)
# %% -------------------------
print("[Step 2] Joining PCG signals with manual labels (collision-safe)...")
df2 = pd.merge(
    pcg_df[[PCG_ID_COL, PCG_POINT_COL, "Normalized_Point", PCG_SIGNAL_COL]].copy(),
    manual_df,
    left_on=[PCG_ID_COL, "Normalized_Point"],
    right_on=["ID", "Normalized_Point"],
    how="inner",
)
print(f"  joined rows: {len(df2)}")

# Sanity checks: ensure no PCG_x/PCG_y
if "PCG_x" in df2.columns or "PCG_y" in df2.columns:
    raise RuntimeError(
        "Found PCG_x/PCG_y after merge; collision fix not applied correctly. "
        f"Columns: {list(df2.columns)}"
    )
if PCG_SIGNAL_COL not in df2.columns:
    raise RuntimeError(f"Signal column '{PCG_SIGNAL_COL}' missing after merge.")
if Y_MANUAL_COL_RENAMED not in df2.columns:
    raise RuntimeError(f"Manual label column '{Y_MANUAL_COL_RENAMED}' missing after merge.")

# %% -------------------------
# %% Preprocess PCG (bandpass) BEFORE SQI extraction
# %% -------------------------
if "pplib" not in globals():
    raise ImportError("preprocessing_lib (pplib) not available. Fix imports to run preprocessing.")

print("\n[Step 2] Preprocessing PCG (Butterworth bandpass)...")
df2[PCG_SIGNAL_COL] = df2[PCG_SIGNAL_COL].apply(
    lambda data: pplib.butterworth_filter(
        data,
        filter_topology="bandpass",
        order=PCG_BPF_ORDER,
        fs=PCG_FS,
        fc=PCG_BPF_FC,
    )
)

# %% -------------------------
# %% Build 3-class target from manual annotations (now in PCG_manual)
# %% -------------------------
df2[Y_MANUAL_COL_RENAMED] = pd.to_numeric(df2[Y_MANUAL_COL_RENAMED], errors="coerce")
df2 = df2.dropna(subset=[Y_MANUAL_COL_RENAMED]).copy()

df2["y_3class_str"] = df2[Y_MANUAL_COL_RENAMED].astype(int).map(RELABEL_MAP)
df2 = df2.dropna(subset=["y_3class_str"]).copy()

le2 = LabelEncoder()
le2.fit(CLASS_ORDER)
df2["y_3class"] = le2.transform(df2["y_3class_str"].values)

print("\n[Step 2] 3-class distribution:")
print(df2["y_3class_str"].value_counts())

# %% -------------------------
# %% Extract ALL unimodal PCG SQIs for each signal (pcg_unimodal_sqi_test-style)
# %% -------------------------
def _safe_float(v):
    """Convert to finite float; otherwise return NaN."""
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def extract_all_pcg_sqi_features(pcg_sig, fs=PCG_FS):
    """
    Extract all unimodal PCG SQIs for one signal.
    Each SQI is isolated in its own try/except so partial results survive.
    """
    # Basic signal sanity
    try:
        x = np.asarray(pcg_sig, dtype=float).squeeze()
        if x.ndim != 1 or len(x) < int(0.5 * fs):
            return {k: np.nan for k in FEATURE_COLS_STEP2}
    except Exception:
        return {k: np.nan for k in FEATURE_COLS_STEP2}

    out = {}

    try:
        out["seSQI"] = _safe_float(sqi_pcg_lib.se_sqi_pcg(x, fs, M=SE_M, r=SE_R))
    except Exception:
        out["seSQI"] = np.nan

    try:
        out["cpSQI"] = _safe_float(sqi_pcg_lib.correlation_prominence_pcg(x, fs))
    except Exception:
        out["cpSQI"] = np.nan

    try:
        out["pr100_200SQI"] = _safe_float(sqi_pcg_lib.pcg_power_ratio_100_200(x, fs))
    except Exception:
        out["pr100_200SQI"] = np.nan

    try:
        out["pr200_400SQI"] = _safe_float(sqi_pcg_lib.pcg_power_ratio_200_400(x, fs))
    except Exception:
        out["pr200_400SQI"] = np.nan

    try:
        out["mean_133_267"] = _safe_float(sqi_pcg_lib.mfcc_mean_133_267_pcg(x, fs))
    except Exception:
        out["mean_133_267"] = np.nan

    try:
        out["median_133_267"] = _safe_float(sqi_pcg_lib.mfcc_median_133_267_pcg(x, fs))
    except Exception:
        out["median_133_267"] = np.nan

    try:
        out["max_600_733"] = _safe_float(sqi_pcg_lib.mfcc_max_600_733_pcg(x, fs))
    except Exception:
        out["max_600_733"] = np.nan

    try:
        out["diff_peak_sqi"] = _safe_float(sqi_pcg_lib.pcg_periodogram_peak_difference(x, fs))
    except Exception:
        out["diff_peak_sqi"] = np.nan

    try:
        out["svdSQI"] = _safe_float(sqi_pcg_lib.svd_sqi_pcg(x, fs, hr_range_bpm=SVD_HR_RANGE_BPM))
    except Exception:
        out["svdSQI"] = np.nan

    return out

print("\n[Step 2] Extracting ALL PCG SQIs for all joined signals...")
feat_df = df2[PCG_SIGNAL_COL].apply(lambda sig: pd.Series(extract_all_pcg_sqi_features(sig, fs=PCG_FS)))
df2 = pd.concat([df2.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

# Drop rows with missing SQI values (strict policy)
n_before = len(df2)
df2 = df2.dropna(subset=FEATURE_COLS_STEP2).copy()
print(f"[Step 2] Rows before dropping NaNs in SQIs: {n_before}, after: {len(df2)}")

# Save extracted features (audit/debug)
features_csv = os.path.join(OUT_DIR_STEP2, "step2_pcg_unimodal_features_extracted.csv")
df2[["ID", "Auscultation_Point_x", "y_3class_str"] + FEATURE_COLS_STEP2].to_csv(features_csv, index=False)
print(f"[Step 2] Saved extracted features: {features_csv}")

# %% -------------------------
# %% ML: Multinomial Logistic Regression (3-class), using ALL SQIs
# %% -------------------------
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

print("\n[Step 2] Train/test split (stratified) and evaluation...")
Xtr2, Xte2, ytr2, yte2 = train_test_split(
    X2, y2,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y2
)

# evaluate_test_split writes into global OUT_DIR -> temporarily point it to step2 folder
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

print("\n[Step 2] Cross-validated summary metrics...")
res2_cv = cross_validated_summary(
    pipeline=lr_pipe_step2,
    X=X2,
    y=y2,
    class_labels=np.array([0, 1, 2], dtype=int),
    task_name="Step 2 — LogReg (PCG unimodal ALL SQIs) — 3-class"
)

# restore OUT_DIR
OUT_DIR = _OUT_DIR_BACKUP

# Save summary table
summary2_df = pd.DataFrame([
    {
        "experiment": "step2_logreg_pcg_unimodal_allSQI_3class_test",
        "n_samples": int(len(y2)),
        "n_features": int(X2.shape[1]),
        "auc_ovr": res2_test["auc_ovr"],
        "accuracy": res2_test["accuracy"],
        "macro_sensitivity": res2_test["macro_sensitivity"],
        "macro_specificity": res2_test["macro_specificity"],
        "macro_f1": res2_test["macro_f1"],
    },
    {
        "experiment": "step2_logreg_pcg_unimodal_allSQI_3class_cv",
        "n_samples": int(len(y2)),
        "n_features": int(X2.shape[1]),
        "auc_ovr": res2_cv["cv_auc_mean"],
        "accuracy": res2_cv["cv_acc_mean"],
        "macro_sensitivity": res2_cv["cv_rec_mean"],
        "macro_specificity": res2_cv["cv_spec_mean"],
        "macro_f1": res2_cv["cv_f1_mean"],
    }
])

summary2_csv = os.path.join(OUT_DIR_STEP2, "step2_logreg_pcg_unimodal_allSQI_3class_summary.csv")
summary2_df.to_csv(summary2_csv, index=False)

print("\n" + "=" * 80)
print("[Step 2 COMPLETE] SUMMARY TABLE")
print("=" * 80)
print(summary2_df.to_string(index=False))
print(f"\nSaved: {summary2_csv}")
print(f"Artifacts folder: {OUT_DIR_STEP2}")


# %% =========================
# %% NEXT STEPS PLACEHOLDER
# %% =========================
# Step 3: ECG unimodal SQIs (all) + LogReg (target from manual Excel column C header 'ECG')
# Step 4: Repeat Step 1–3 with SVM
# Step 5: Aggregate all results into a final comparison report
