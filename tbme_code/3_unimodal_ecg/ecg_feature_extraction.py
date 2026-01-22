# -*- coding: utf-8 -*-
"""
Unimodal ECG feature extraction (SQIs)

Purpose
-------
Extract a fixed set of ECG Signal Quality Indices (SQIs) from ECG signals and export a CSV with:

    [ID, Auscultation_Point, ...ecg_sqis..., manual_label_only_ecg]

Manual labels
-------------
Manual ECG labels are read from the Excel file 'ulsge_manual_sqa.xlsx', which lives one
folder ABOVE the Multimodal_SQA project root.

The manual ECG label is mapped into the project 3-class convention:
    0           -> low_quality
    1, 2        -> uncertain
    3, 4, 5     -> high_quality

Alignment / merging
-------------------
This step guarantees that each exported row corresponds to an ECG signal that is correctly
aligned with its manual ECG annotation, using the project-wide matching rule:
  - ID == Trial  (as str)
  - Auscultation_Point == Spot (ignoring underscores, case-insensitive)

Implementation detail:
- We first build an aligned table that contains:
    ["ID", "Auscultation_Point", "ECG_manual", "ECG_signal"]
  guaranteeing the manual label and the signal correspond to the same record.
- Only after alignment is guaranteed, we preprocess ECG_signal and extract SQIs.

Signal preprocessing
--------------------
Butterworth lowpass filter:
  - fs = 500 Hz
  - order = 4
  - fc = 125 Hz

SQIs extracted
--------------
FEATURE_COLS = [
    "bSQI",
    "pSQI",
    "sSQI",
    "kSQI",
    "fSQI",
    "basSQI",
]

Outputs
-------
- step3_ecg_unimodal_features_extracted.csv

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
# File location: .../Multimodal_SQA/tbme_code/3_unimodal_ecg/ecg_feature_extraction.py
# Project "root" for imports is Multimodal_SQA (two levels above this file).
# Manual labels live one folder ABOVE Multimodal_SQA.
# ---------------------------------------------------------------------
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # Multimodal_SQA
parent_path = os.path.abspath(os.path.join(root_path, ".."))  # folder above Multimodal_SQA
if root_path not in sys.path:
    sys.path.append(root_path)

# Project libraries (resolved via sys.path append above)
import preprocessing_lib as pplib
import sqi_ecg_lib

warnings.filterwarnings("ignore", category=UserWarning)

# %% Global config

# ECG signals pickle is in DatasetCHVNGE folder (one folder above Multimodal_SQA, per project layout)
ECG_PKL_PATH = os.path.join(parent_path, "DatasetCHVNGE", "ecg_ulsge.pkl")

# Manual labels excel is one folder above Multimodal_SQA
MQ_PATH = os.path.join(parent_path, "ulsge_manual_sqa.xlsx")

# Output directory (placed next to this script)
OUT_DIR = os.path.join(os.path.dirname(__file__), "exp_step3_ecg_features")
os.makedirs(OUT_DIR, exist_ok=True)

# --- ECG preprocessing parameters ---
ECG_FS = 500
ECG_LPF_ORDER = 4
ECG_LPF_FC = 125  # Hz

# --- Expected ECG dataframe schema ---
ECG_ID_COL = "ID"
ECG_POINT_COL = "Auscultation_Point"
ECG_SIGNAL_COL = "ECG"  # raw waveform
# (Optional: pickle may also have 'Source', not used here)

# --- SQI feature list (output columns) ---
FEATURE_COLS = [
    "bSQI",
    "pSQI",
    "sSQI",
    "kSQI",
    "fSQI",
    "basSQI",
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


def merge_ecg_manual_signals(ecg_df: pd.DataFrame, m_quality: pd.DataFrame) -> pd.DataFrame:
    """
    Align manual ECG labels to ECG signals and return an aligned table with signal payload.

    Join rule (project convention)
    ------------------------------
      - ecg_df.ID (string) matches m_quality.Trial (string)
      - ecg_df.Auscultation_Point matches m_quality.Spot after normalization:
            remove '_' and uppercase.

    Output
    ------
    DataFrame with:
      ["ID", "Auscultation_Point", "ECG_manual", "ECG_signal"]

    Notes
    -----
    - Auscultation_Point is taken from the signal-side dataframe to ensure the exported
      key matches the signal record that will be processed.
    - ECG_signal is taken from ecg_df[ECG_SIGNAL_COL].
    - The function is intentionally strict: if required columns are missing, it raises.
    """
    # Validate required columns (fail fast)
    for col in [ECG_ID_COL, ECG_POINT_COL, ECG_SIGNAL_COL]:
        if col not in ecg_df.columns:
            raise ValueError(
                f"ECG dataframe missing required column '{col}'. "
                f"Found columns: {list(ecg_df.columns)}"
            )

    for col in ["Trial", "Spot", "ECG"]:
        if col not in m_quality.columns:
            raise ValueError(
                f"Manual labels missing required column '{col}'. "
                f"Found columns: {list(m_quality.columns)}"
            )

    # Signal-side keys + payload
    sig = ecg_df[[ECG_ID_COL, ECG_POINT_COL, ECG_SIGNAL_COL]].copy()
    sig[ECG_ID_COL] = sig[ECG_ID_COL].astype(str)
    sig["_k_point"] = _norm_point(sig[ECG_POINT_COL])

    # Manual-side keys + label
    man = m_quality[["Trial", "Spot", "ECG"]].copy()
    man["Trial"] = man["Trial"].astype(str)
    man["_k_spot"] = _norm_point(man["Spot"])

    # Rename manual columns to avoid suffixing surprises after merge
    man = man.rename(columns={"Trial": "ID_manual", "ECG": "ECG_manual"})

    # Merge: guarantee alignment between manual annotations and signal records
    merged = pd.merge(
        sig[[ECG_ID_COL, ECG_POINT_COL, ECG_SIGNAL_COL, "_k_point"]].copy(),
        man[["ID_manual", "_k_spot", "ECG_manual"]].copy(),
        left_on=[ECG_ID_COL, "_k_point"],
        right_on=["ID_manual", "_k_spot"],
        how="inner",
    )

    # Rename to final schema (signal-side keys are authoritative)
    merged = merged.rename(
        columns={
            ECG_ID_COL: "ID",
            ECG_POINT_COL: "Auscultation_Point",
            ECG_SIGNAL_COL: "ECG_signal",
        }
    )

    # Drop helper keys and the manual-side join ID
    merged = merged.drop(columns=["_k_point", "_k_spot", "ID_manual"])

    return merged[["ID", "Auscultation_Point", "ECG_manual", "ECG_signal"]].copy()


# %% SQI extraction utilities


def _safe_float(v):
    """Convert to float, returning NaN on failure or non-finite values."""
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def extract_all_ecg_sqi_features(ecg_sig, fs=ECG_FS) -> dict:
    """
    Extract all ECG SQIs for one ECG signal.

    Robustness
    ----------
    - Invalid signal shape/type -> all NaNs
    - Very short signals (<0.5 s) -> all NaNs
    - Individual SQI failures -> NaN only for that SQI

    Returns
    -------
    dict mapping FEATURE_COLS -> float/NaN
    """
    try:
        x = np.asarray(ecg_sig, dtype=float).squeeze()
        if x.ndim != 1 or len(x) < int(0.5 * fs):
            return {k: np.nan for k in FEATURE_COLS}
    except Exception:
        return {k: np.nan for k in FEATURE_COLS}

    out = {}

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


# %% Load data
print("\nLoading ECG signals...")
ecg_df = pd.read_pickle(ECG_PKL_PATH).copy()

# Fail fast if schema differs from expected
expected_cols = [ECG_ID_COL, ECG_POINT_COL, ECG_SIGNAL_COL]
for c in expected_cols:
    if c not in ecg_df.columns:
        raise ValueError(f"ecg_ulsge.pkl missing '{c}'. Found columns: {list(ecg_df.columns)}")

print("Loading manual labels...")
m_quality = pd.read_excel(MQ_PATH).copy()

# %% Align manual ECG labels and signals into a single aligned table
print("Aligning manual ECG labels with ECG signals...")
df3 = merge_ecg_manual_signals(ecg_df, m_quality)
print(f"  aligned rows (signal + manual): {len(df3)}")

# %% Preprocess ECG signals (only after alignment is guaranteed)
print("\nPreprocessing ECG (lowpass Butterworth)...")
df3["ECG_signal"] = df3["ECG_signal"].apply(
    lambda data: pplib.butterworth_filter(
        data,
        filter_topology="lowpass",
        order=ECG_LPF_ORDER,
        fs=ECG_FS,
        fc=ECG_LPF_FC,
    )
)

# %% Build 3-class manual label (ECG only)
df3["ECG_manual"] = pd.to_numeric(df3["ECG_manual"], errors="coerce")
df3 = df3.dropna(subset=["ECG_manual"]).copy()

df3["manual_label_only_ecg"] = df3["ECG_manual"].astype(int).map(RELABEL_MAP)
df3 = df3.dropna(subset=["manual_label_only_ecg"]).copy()

print("\n3-class distribution (manual_label_only_ecg):")
print(df3["manual_label_only_ecg"].value_counts())

# %% Extract ECG SQIs
print("\nExtracting ECG SQIs...")
feat_df = df3["ECG_signal"].apply(lambda sig: pd.Series(extract_all_ecg_sqi_features(sig, fs=ECG_FS)))
df3 = pd.concat([df3.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

# Drop rows that have any missing SQI (strict feature completeness)
n_before = len(df3)
df3 = df3.dropna(subset=FEATURE_COLS).copy()
print(f"Rows before dropping NaNs in SQIs: {n_before}, after: {len(df3)}")

# %% Save output CSV
# Required structure: [ID, Auscultation_Point, ...ecg_sqis..., manual_label_only_ecg]
out_cols = ["ID", "Auscultation_Point"] + FEATURE_COLS + ["manual_label_only_ecg"]
features_csv = os.path.join(OUT_DIR, "step3_ecg_unimodal_features_extracted.csv")

df3[out_cols].to_csv(features_csv, index=False)
print(f"Saved extracted features: {features_csv}")
