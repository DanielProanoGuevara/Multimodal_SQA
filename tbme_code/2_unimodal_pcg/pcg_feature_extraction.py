# -*- coding: utf-8 -*-
"""
Unimodal PCG feature extraction (SQIs)

Purpose
-------
Extract a fixed set of PCG Signal Quality Indices (SQIs) from PCG signals and export a CSV with:

    [ID, Auscultation_Point, ...pcg_sqis..., manual_label_only_pcg]

Manual labels
-------------
Manual PCG labels are read from the Excel file 'ulsge_manual_sqa.xlsx', which lives one
folder ABOVE the Multimodal_SQA project root.

The manual PCG label is mapped into the project 3-class convention:
    0           -> low_quality
    1, 2        -> uncertain
    3, 4, 5     -> high_quality

Alignment / merging
-------------------
This step guarantees that each exported row corresponds to a PCG signal that is correctly
aligned with its manual PCG annotation, using the project-wide matching rule:
  - ID == Trial  (as str)
  - Auscultation_Point == Spot (ignoring underscores, case-insensitive)

Implementation detail:
- We first build an aligned table that contains:
    ["ID", "Auscultation_Point", "PCG_manual", "PCG_signal"]
  guaranteeing the manual label and the signal correspond to the same record.
- Only after alignment is guaranteed, we preprocess PCG_signal and extract SQIs.

Signal preprocessing
--------------------
Bandpass Butterworth filter:
  - fs = 3000 Hz
  - order = 4
  - fc = [50, 250] Hz

SQIs extracted
--------------
FEATURE_COLS = [
    "seSQI", "cpSQI", "pr100_200SQI", "pr200_400SQI",
    "mean_133_267", "median_133_267", "max_600_733",
    "diff_peak_sqi", "svdSQI"
]

Outputs
-------
- step2_pcg_unimodal_features_extracted.csv

@author: Daniel Proaño-Guevara
"""

# %% Imports
import os
import sys
import warnings
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project import path handling:
# File location: .../Multimodal_SQA/tbme_code/2_unimodal_pcg/pcg_feature_extraction.py
# Project "root" for imports is Multimodal_SQA (two levels above this file).
# Manual labels live one folder ABOVE Multimodal_SQA.
# ---------------------------------------------------------------------
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))        # Multimodal_SQA
parent_path = os.path.abspath(os.path.join(root_path, ".."))                           # folder above Multimodal_SQA
if root_path not in sys.path:
    sys.path.append(root_path)

# Project libraries (resolved via sys.path append above)
import preprocessing_lib as pplib
import sqi_pcg_lib

warnings.filterwarnings("ignore", category=UserWarning)

# %% Global config

# PCG signals pickle is inside Multimodal_SQA (DatasetCHVNGE folder)
PCG_PKL_PATH = os.path.join(parent_path, "DatasetCHVNGE", "pcg_ulsge.pkl")

# Manual labels excel is one folder above Multimodal_SQA
MQ_PATH = os.path.join(parent_path, "ulsge_manual_sqa.xlsx")

# Output directory (placed next to this script)
OUT_DIR = os.path.join(os.path.dirname(__file__), "exp_step2_pcg_features")
os.makedirs(OUT_DIR, exist_ok=True)

# --- PCG preprocessing parameters ---
PCG_FS = 3000
PCG_BPF_ORDER = 4
PCG_BPF_FC = [50, 250]

# --- SQI parameters used by some metrics ---
SE_M = 2
SE_R = 0.0008
SVD_HR_RANGE_BPM = (70, 220)

# --- Expected PCG dataframe schema ---
PCG_ID_COL = "ID"
PCG_POINT_COL = "Auscultation_Point"
PCG_SIGNAL_COL = "PCG"  # raw waveform
# (Optional: the pickle also has 'Source', not used in this step)

# --- SQI feature list (output columns) ---
FEATURE_COLS = [
    "seSQI", "cpSQI", "pr100_200SQI", "pr200_400SQI",
    "mean_133_267", "median_133_267", "max_600_733",
    "diff_peak_sqi", "svdSQI"
]

# --- Manual label remapping to 3 classes (project convention) ---
RELABEL_MAP = {
    0: "low_quality",
    1: "uncertain",
    2: "uncertain",
    3: "high_quality",
    4: "high_quality",
    5: "high_quality",
}

# %% Merge utilities


def _norm_point(series: pd.Series) -> pd.Series:
    """Normalize auscultation points: remove '_' and uppercase (for robust joins)."""
    return series.astype(str).str.replace("_", "", regex=False).str.upper()


def merge_pcg_manual_signals(pcg_df: pd.DataFrame, m_quality: pd.DataFrame) -> pd.DataFrame:
    """
    Align manual PCG labels to PCG signals and return an aligned table with signal payload.

    Output:
      ["ID", "Auscultation_Point", "PCG_manual", "PCG_signal"]
    """
    # Validate required columns (fail fast)
    for col in [PCG_ID_COL, PCG_POINT_COL, PCG_SIGNAL_COL]:
        if col not in pcg_df.columns:
            raise ValueError(
                f"PCG dataframe missing required column '{col}'. "
                f"Found columns: {list(pcg_df.columns)}"
            )
    for col in ["Trial", "Spot", "PCG"]:
        if col not in m_quality.columns:
            raise ValueError(
                f"Manual labels missing required column '{col}'. "
                f"Found columns: {list(m_quality.columns)}"
            )

    # Signal-side keys + payload
    sig = pcg_df[[PCG_ID_COL, PCG_POINT_COL, PCG_SIGNAL_COL]].copy()
    sig[PCG_ID_COL] = sig[PCG_ID_COL].astype(str)
    sig["_k_point"] = _norm_point(sig[PCG_POINT_COL])

    # Manual-side keys + label
    man = m_quality[["Trial", "Spot", "PCG"]].copy()
    man["Trial"] = man["Trial"].astype(str)
    man["_k_spot"] = _norm_point(man["Spot"])

    # Rename manual columns to avoid any suffixing surprises
    # (we want fully deterministic column names after merge)
    man = man.rename(columns={"Trial": "ID_manual", "PCG": "PCG_manual"})

    merged = pd.merge(
        sig[[PCG_ID_COL, PCG_POINT_COL, PCG_SIGNAL_COL, "_k_point"]].copy(),
        man[["ID_manual", "_k_spot", "PCG_manual"]].copy(),
        left_on=[PCG_ID_COL, "_k_point"],
        right_on=["ID_manual", "_k_spot"],
        how="inner",
    )

    # Rename to final schema (signal-side keys are authoritative)
    merged = merged.rename(
        columns={
            PCG_ID_COL: "ID",
            PCG_POINT_COL: "Auscultation_Point",
            PCG_SIGNAL_COL: "PCG_signal",
        }
    )

    # Drop helper keys and the manual-side join ID
    merged = merged.drop(columns=["_k_point", "_k_spot", "ID_manual"])

    return merged[["ID", "Auscultation_Point", "PCG_manual", "PCG_signal"]].copy()


# %% SQI extraction utilities


def _safe_float(v):
    """Convert to float, returning NaN on failure or non-finite values."""
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def extract_all_pcg_sqi_features(pcg_sig, fs=PCG_FS) -> dict:
    """
    Extract all PCG SQIs for one PCG signal.

    Robustness
    ----------
    - Invalid signal shape/type -> all NaNs
    - Very short signals (<0.5 s) -> all NaNs
    - Individual SQI failures -> NaN only for that SQI
    """
    try:
        x = np.asarray(pcg_sig, dtype=float).squeeze()
        if x.ndim != 1 or len(x) < int(0.5 * fs):
            return {k: np.nan for k in FEATURE_COLS}
    except Exception:
        return {k: np.nan for k in FEATURE_COLS}

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


# %% Load data
print("\nLoading PCG signals...")
pcg_df = pd.read_pickle(PCG_PKL_PATH).copy()

# Fail fast if schema differs from expected
expected_cols = [PCG_ID_COL, PCG_POINT_COL, PCG_SIGNAL_COL]
for c in expected_cols:
    if c not in pcg_df.columns:
        raise ValueError(f"pcg_ulsge.pkl missing '{c}'. Found columns: {list(pcg_df.columns)}")

print("Loading manual labels...")
m_quality = pd.read_excel(MQ_PATH).copy()

# %% Align manual PCG labels and signals into a single aligned table
print("Aligning manual PCG labels with PCG signals...")
df2 = merge_pcg_manual_signals(pcg_df, m_quality)
print(f"  aligned rows (signal + manual): {len(df2)}")

# %% Preprocess PCG signals (only after alignment is guaranteed)
print("\nPreprocessing PCG (bandpass Butterworth)...")
df2["PCG_signal"] = df2["PCG_signal"].apply(
    lambda data: pplib.butterworth_filter(
        data,
        filter_topology="bandpass",
        order=PCG_BPF_ORDER,
        fs=PCG_FS,
        fc=PCG_BPF_FC,
    )
)

# %% Build 3-class manual label (PCG only)
df2["PCG_manual"] = pd.to_numeric(df2["PCG_manual"], errors="coerce")
df2 = df2.dropna(subset=["PCG_manual"]).copy()

df2["manual_label_only_pcg"] = df2["PCG_manual"].astype(int).map(RELABEL_MAP)
df2 = df2.dropna(subset=["manual_label_only_pcg"]).copy()

print("\n3-class distribution (manual_label_only_pcg):")
print(df2["manual_label_only_pcg"].value_counts())

# %% Extract PCG SQIs
print("\nExtracting PCG SQIs...")
feat_df = df2["PCG_signal"].apply(lambda sig: pd.Series(extract_all_pcg_sqi_features(sig, fs=PCG_FS)))
df2 = pd.concat([df2.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

# Drop rows that have any missing SQI (strict feature completeness)
n_before = len(df2)
df2 = df2.dropna(subset=FEATURE_COLS).copy()
print(f"Rows before dropping NaNs in SQIs: {n_before}, after: {len(df2)}")

# %% Save output CSV
out_cols = ["ID", "Auscultation_Point"] + FEATURE_COLS + ["manual_label_only_pcg"]
features_csv = os.path.join(OUT_DIR, "step2_pcg_unimodal_features_extracted.csv")

df2[out_cols].to_csv(features_csv, index=False)
print(f"Saved extracted features: {features_csv}")
