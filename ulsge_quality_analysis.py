# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:16:31 2024

@author: danie
"""

# %%
# Imports

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as pltb
import preprocessing_lib as pplib
import feature_extraction_lib as ftelib
import file_process_lib as importlib
from sklearn.preprocessing import OneHotEncoder

import pickle
import copy

# %% Constants
AVERAGE_WINDOW = 3

# %% Import PCG
root_dir = r'..\DatasetCHVNGE\pcg_ulsge.pkl'
pcg_df_original = pd.read_pickle(root_dir)
# Deep copy
pcg_df = copy.deepcopy(pcg_df_original)
# Resample them to 50 Hz
pcg_df['PCG'] = pcg_df['PCG'].apply(
    lambda data: pplib.downsample(data, 3000, 1000))

# Import Predictions
pred_path = r'..\ulsge_pcg_pred.pkl'
with open(pred_path, 'rb') as file:
    pcg_predictions = pickle.load(file)

# Smooth Predictions
pcg_processed_predictions = [
    np.column_stack([pplib.moving_average(data[:, i], AVERAGE_WINDOW)
                    for i in range(4)])
    for data in pcg_predictions
]

# Reverse one-hot


def max_temporal_modelling(seq, num_states=4):
    for t in range(1, len(seq)):
        if seq[t] != seq[t-1] and seq[t] != ((seq[t-1] + 1) % num_states):
            seq[t] = seq[t-1]
    return seq


pcg_pred_labels = [ftelib.reverse_one_hot_encoding(
    pred) for pred in pcg_processed_predictions]
pcg_prediction_labels = copy.deepcopy(pcg_pred_labels)
# pcg_state_predictions = np.array(
#     [max_temporal_modelling(prediction) for prediction in pcg_prediction_labels], dtype=object)
pcg_state_predictions = np.array(
    [prediction for prediction in pcg_prediction_labels], dtype=object)


# %% Import ECG
root_dir = r'..\DatasetCHVNGE\ecg_ulsge.pkl'
ecg_df_original = pd.read_pickle(root_dir)
# Deep copy
ecg_df = copy.deepcopy(ecg_df_original)
# Resample them to 50 Hz
ecg_df['ECG'] = ecg_df['ECG'].apply(
    lambda data: pplib.upsample(data, 500, 1000))
# Import Predictions
predictions_pickle_path = r'..\ulsge_ecg_pred.pkl'
with open(predictions_pickle_path, 'rb') as file:
    ecg_predictions = pickle.load(file)

# Smooth Predictions
ecg_processed_predictions = [
    np.column_stack([pplib.moving_average(data[:, i], AVERAGE_WINDOW)
                    for i in range(4)])
    for data in ecg_predictions
]

# Reverse one-hot
ecg_pred_labels = [ftelib.reverse_one_hot_encoding(
    pred, desired_order=[1, 3, 0, 2]) for pred in ecg_processed_predictions]
ecg_prediction_labels = copy.deepcopy(ecg_pred_labels)
ecg_state_predictions = np.array(
    [prediction for prediction in ecg_prediction_labels], dtype=object)


# %% Scoring Calculations base functions


# Extract intervals from the delineation sequences

def get_intervals(signal, target_label, min_duration=1):
    """
    Extract intervals (start, end indices) where signal == target_label.
    Only intervals with duration >= min_duration are returned.
    """
    intervals = []
    in_interval = False
    start = None
    for i, val in enumerate(signal):
        if val == target_label and not in_interval:
            in_interval = True
            start = i
        elif val != target_label and in_interval:
            end = i  # end is the first index where value changes
            if (end - start) >= min_duration:
                intervals.append((start, end))
            in_interval = False
    # If the signal ends while still in an interval:
    if in_interval and (len(signal) - start) >= min_duration:
        intervals.append((start, len(signal)))
    return intervals

# If the intervals match in at least one point, the overlap time is presented


def compute_overlap(interval_a, interval_b):
    """
    Compute the overlap between two intervals.
    Each interval is a tuple (start, end).
    """
    start_a, end_a = interval_a
    start_b, end_b = interval_b
    overlap = max(0, min(end_a, end_b) - max(start_a, start_b))
    return overlap


def match_intervals(ref_intervals, test_intervals, min_overlap=1):
    """
    For each reference interval, check if there is any test interval with
    at least min_overlap samples overlapping.
    Returns the number of matches.
    """
    matches = 0
    for ref in ref_intervals:
        matched = False
        for test in test_intervals:
            if compute_overlap(ref, test) >= min_overlap:
                matched = True
                break
        if matched:
            matches += 1
    return matches


def alignment_metric_min_lin(ecg_signal, pcg_signal, lambda_penalty=0.5, min_duration=1, min_overlap=1, extend_window=8):
    """
    Compute the bidirectional alignment metric for the fiducial segments.
    Assumes:
      - ECG: QRS (0) and T-wave (2)
      - PCG: S1 (0) and S2 (2)
    Returns the overall metric and individual directional scores.
    """
    # Extract intervals for physiologically valid segments
    ecg_qrs = get_intervals(ecg_signal, target_label=0,
                            min_duration=min_duration)
    ecg_twave = get_intervals(
        ecg_signal, target_label=2, min_duration=min_duration)
    pcg_s1 = get_intervals(pcg_signal, target_label=0,
                           min_duration=min_duration)
    pcg_s2 = get_intervals(pcg_signal, target_label=2,
                           min_duration=min_duration)

    # ECG-to-PCG matching
    # Extend to the right ECG segments
    qrs_extended = pplib.extend_intervals(ecg_qrs, 'right', extend_window)
    twave_extended = pplib.extend_intervals(ecg_twave, 'right', extend_window)
    match_qrs = match_intervals(qrs_extended, pcg_s1, min_overlap=min_overlap)
    match_twave = match_intervals(
        twave_extended, pcg_s2, min_overlap=min_overlap)
    # QRS should match with S1, T-wave should match with S2
    total_ecg = len(ecg_qrs) + len(ecg_twave)
    matches_ecg = match_qrs + match_twave
    # Apply linear penalty for missing matches
    score_ecg_to_pcg = (matches_ecg - lambda_penalty *
                        (total_ecg - matches_ecg)) / total_ecg if total_ecg > 0 else 0

    # PCG-to-ECG matching
    # Extend PCG segments to the left
    s1_extended = pplib.extend_intervals(pcg_s1, 'left', extend_window)
    s2_extended = pplib.extend_intervals(pcg_s2, 'left', extend_window)
    match_s1 = match_intervals(s1_extended, ecg_qrs, min_overlap=min_overlap)
    match_s2 = match_intervals(s2_extended, ecg_twave, min_overlap=min_overlap)
    # S1 should match with QRS, S2 should match with T-wave
    total_pcg = len(pcg_s1) + len(pcg_s2)
    matches_pcg = match_s1 + match_s2
    score_pcg_to_ecg = (matches_pcg - lambda_penalty *
                        (total_pcg - matches_pcg)) / total_pcg if total_pcg > 0 else 0

    # Overall metric is the minima of both directional scores
    overall_metric = min(score_ecg_to_pcg, score_pcg_to_ecg)
    if overall_metric < 0:
        overall_metric = 0
    return overall_metric


def alignment_metric_avg_lin(ecg_signal, pcg_signal, lambda_penalty=0.5, min_duration=1, min_overlap=1, extend_window=8):
    """
    Compute the bidirectional alignment metric for the fiducial segments.
    Assumes:
      - ECG: QRS (0) and T-wave (2)
      - PCG: S1 (0) and S2 (2)
    Returns the overall metric and individual directional scores.
    """
    # Extract intervals for physiologically valid segments
    ecg_qrs = get_intervals(ecg_signal, target_label=0,
                            min_duration=min_duration)
    ecg_twave = get_intervals(
        ecg_signal, target_label=2, min_duration=min_duration)
    pcg_s1 = get_intervals(pcg_signal, target_label=0,
                           min_duration=min_duration)
    pcg_s2 = get_intervals(pcg_signal, target_label=2,
                           min_duration=min_duration)

    # ECG-to-PCG matching
    # Extend to the right ECG segments
    qrs_extended = pplib.extend_intervals(ecg_qrs, 'right', extend_window)
    twave_extended = pplib.extend_intervals(ecg_twave, 'right', extend_window)
    match_qrs = match_intervals(qrs_extended, pcg_s1, min_overlap=min_overlap)
    match_twave = match_intervals(
        twave_extended, pcg_s2, min_overlap=min_overlap)
    # QRS should match with S1, T-wave should match with S2
    total_ecg = len(ecg_qrs) + len(ecg_twave)
    matches_ecg = match_qrs + match_twave
    # Apply linear penalty for missing matches
    score_ecg_to_pcg = (matches_ecg - lambda_penalty *
                        (total_ecg - matches_ecg)) / total_ecg if total_ecg > 0 else 0

    # PCG-to-ECG matching
    # Extend PCG segments to the left
    s1_extended = pplib.extend_intervals(pcg_s1, 'left', extend_window)
    s2_extended = pplib.extend_intervals(pcg_s2, 'left', extend_window)
    match_s1 = match_intervals(s1_extended, ecg_qrs, min_overlap=min_overlap)
    match_s2 = match_intervals(s2_extended, ecg_twave, min_overlap=min_overlap)
    # S1 should match with QRS, S2 should match with T-wave
    total_pcg = len(pcg_s1) + len(pcg_s2)
    matches_pcg = match_s1 + match_s2
    score_pcg_to_ecg = (matches_pcg - lambda_penalty *
                        (total_pcg - matches_pcg)) / total_pcg if total_pcg > 0 else 0

    # Overall metric is the average of both directional scores
    overall_metric = 0.5 * (score_ecg_to_pcg + score_pcg_to_ecg)
    if overall_metric < 0:
        overall_metric = 0
    return overall_metric


def alignment_metric_min_min(ecg_signal, pcg_signal, lambda_penalty=0.1, min_duration=1, min_overlap=1, extend_window=8):
    """
    Compute the bidirectional alignment metric for the fiducial segments.
    Assumes:
      - ECG: QRS (0) and T-wave (2)
      - PCG: S1 (0) and S2 (2)
    Returns the overall metric and individual directional scores.
    """
    # Extract intervals for physiologically valid segments
    ecg_qrs = get_intervals(ecg_signal, target_label=0,
                            min_duration=min_duration)
    ecg_twave = get_intervals(
        ecg_signal, target_label=2, min_duration=min_duration)
    pcg_s1 = get_intervals(pcg_signal, target_label=0,
                           min_duration=min_duration)
    pcg_s2 = get_intervals(pcg_signal, target_label=2,
                           min_duration=min_duration)

    # ECG-to-PCG matching
    # Extend to the right ECG segments
    qrs_extended = pplib.extend_intervals(ecg_qrs, 'right', extend_window)
    twave_extended = pplib.extend_intervals(ecg_twave, 'right', extend_window)
    match_qrs = match_intervals(qrs_extended, pcg_s1, min_overlap=min_overlap)
    match_twave = match_intervals(
        twave_extended, pcg_s2, min_overlap=min_overlap)
    # QRS should match with S1, T-wave should match with S2
    # Compute score for QRS: if no QRS intervals, assume worst match (score = 0)
    if len(qrs_extended) > 0:
        score_qrs = (match_qrs - lambda_penalty *
                     (len(ecg_qrs) - match_qrs)) / len(ecg_qrs)
    else:
        score_qrs = 0

    # Compute score for T-wave: if no T-wave intervals, assume worst match (score = 0)
    if len(twave_extended) > 0:
        score_twave = (match_twave - lambda_penalty *
                       (len(ecg_twave) - match_twave)) / len(ecg_twave)
    else:
        score_twave = 0

    # Overall ECG-to-PCG score is the minimum of the two scores to ensure both segments align well
    score_ecg_to_pcg = min(score_qrs, score_twave)

    # PCG-to-ECG matching
    # Extend PCG segments to the left
    s1_extended = pplib.extend_intervals(pcg_s1, 'left', extend_window)
    s2_extended = pplib.extend_intervals(pcg_s2, 'left', extend_window)
    match_s1 = match_intervals(s1_extended, ecg_qrs, min_overlap=min_overlap)
    match_s2 = match_intervals(s2_extended, ecg_twave, min_overlap=min_overlap)
    # S1 should match with QRS, S2 should match with T-wave
    # Compute score for S1: if no S1 intervals, assume worst match (score = 0)
    if len(s1_extended) > 0:
        score_s1 = (match_s1 - lambda_penalty *
                    (len(pcg_s1) - match_s1)) / len(pcg_s1)
    else:
        score_s1 = 0

    # Compute score for S2: if no S2 intervals, assume worst match (score = 0)
    if len(s2_extended) > 0:
        score_s2 = (match_s2 - lambda_penalty *
                    (len(pcg_s2) - match_s2)) / len(pcg_s2)
    else:
        score_s2 = 0

    # Overall PCG-to-ECG score is the minimum of the two scores to ensure both segments align well
    score_pcg_to_ecg = min(score_s1, score_s2)

    # Overall metric is the minima of both directional scores
    overall_metric = min(score_ecg_to_pcg, score_pcg_to_ecg)
    if overall_metric < 0:
        overall_metric = 0
    return overall_metric


def alignment_metric_avg_min(ecg_signal, pcg_signal, lambda_penalty=0.5, min_duration=1, min_overlap=1, extend_window=8):
    """
    Compute the bidirectional alignment metric for the fiducial segments.
    Assumes:
      - ECG: QRS (0) and T-wave (2)
      - PCG: S1 (0) and S2 (2)
    Returns the overall metric and individual directional scores.
    """
    # Extract intervals for physiologically valid segments
    ecg_qrs = get_intervals(ecg_signal, target_label=0,
                            min_duration=min_duration)
    ecg_twave = get_intervals(
        ecg_signal, target_label=2, min_duration=min_duration)
    pcg_s1 = get_intervals(pcg_signal, target_label=0,
                           min_duration=min_duration)
    pcg_s2 = get_intervals(pcg_signal, target_label=2,
                           min_duration=min_duration)

    # ECG-to-PCG matching
    # Extend to the right ECG segments
    qrs_extended = pplib.extend_intervals(ecg_qrs, 'right', extend_window)
    twave_extended = pplib.extend_intervals(ecg_twave, 'right', extend_window)
    match_qrs = match_intervals(qrs_extended, pcg_s1, min_overlap=min_overlap)
    match_twave = match_intervals(
        twave_extended, pcg_s2, min_overlap=min_overlap)
    # QRS should match with S1, T-wave should match with S2
    # Compute score for QRS: if no QRS intervals, assume worst match (score = 0)
    if len(qrs_extended) > 0:
        score_qrs = (match_qrs - lambda_penalty *
                     (len(ecg_qrs) - match_qrs)) / len(ecg_qrs)
    else:
        score_qrs = 0

    # Compute score for T-wave: if no T-wave intervals, assume worst match (score = 0)
    if len(twave_extended) > 0:
        score_twave = (match_twave - lambda_penalty *
                       (len(ecg_twave) - match_twave)) / len(ecg_twave)
    else:
        score_twave = 0

    # Overall ECG-to-PCG score is the minimum of the two scores to ensure both segments align well
    score_ecg_to_pcg = min(score_qrs, score_twave)

    # PCG-to-ECG matching
    # Extend PCG segments to the left
    s1_extended = pplib.extend_intervals(pcg_s1, 'left', extend_window)
    s2_extended = pplib.extend_intervals(pcg_s2, 'left', extend_window)
    match_s1 = match_intervals(s1_extended, ecg_qrs, min_overlap=min_overlap)
    match_s2 = match_intervals(s2_extended, ecg_twave, min_overlap=min_overlap)
    # S1 should match with QRS, S2 should match with T-wave
    # Compute score for S1: if no S1 intervals, assume worst match (score = 0)
    if len(s1_extended) > 0:
        score_s1 = (match_s1 - lambda_penalty *
                    (len(pcg_s1) - match_s1)) / len(pcg_s1)
    else:
        score_s1 = 0

    # Compute score for S2: if no S2 intervals, assume worst match (score = 0)
    if len(s2_extended) > 0:
        score_s2 = (match_s2 - lambda_penalty *
                    (len(pcg_s2) - match_s2)) / len(pcg_s2)
    else:
        score_s2 = 0

    # Overall PCG-to-ECG score is the minimum of the two scores to ensure both segments align well
    score_pcg_to_ecg = min(score_s1, score_s2)

    # Overall metric is the average of both directional scores
    overall_metric = 0.5 * (score_ecg_to_pcg + score_pcg_to_ecg)
    if overall_metric < 0:
        overall_metric = 0
    return overall_metric


# %% Dataset Analysis

# Process all rows in ecg_df and compute metrics for each corresponding ECG and PCG signal

results = []

for idx in range(len(ecg_df)):
    # Retrieve signals for the current index
    ecg_signal = ecg_state_predictions[idx]
    pcg_signal = pcg_state_predictions[idx]

    # Compute alignment metrics using the defined functions
    metric_min_lin = alignment_metric_min_lin(
        ecg_signal, pcg_signal, lambda_penalty=0.1)
    metric_avg_lin = alignment_metric_avg_lin(
        ecg_signal, pcg_signal, lambda_penalty=0.1)
    metric_min_min = alignment_metric_min_min(
        ecg_signal, pcg_signal, lambda_penalty=0.1)
    metric_avg_min = alignment_metric_avg_min(
        ecg_signal, pcg_signal, lambda_penalty=0.1)

    # Retrieve metadata; fallback to index if columns are missing
    row_id = ecg_df.iloc[idx]['ID'] if 'ID' in ecg_df.columns else idx
    ausc_point = ecg_df.iloc[idx]['Auscultation_Point'] if 'Auscultation_Point' in ecg_df.columns else None

    results.append({
        'ID': row_id,
        'Auscultation_Point': ausc_point,
        'alignment_metric_min_lin': metric_min_lin,
        'alignment_metric_avg_lin': metric_avg_lin,
        'alignment_metric_min_min': metric_min_min,
        'alignment_metric_avg_min': metric_avg_min
    })

# Create a DataFrame from the results and export it as a .pkl file
results_df = pd.DataFrame(results)
output_path = r'..\ulsge_quality_metrics.pkl'
results_df.to_pickle(output_path)
print(f"Results exported to {output_path}")
