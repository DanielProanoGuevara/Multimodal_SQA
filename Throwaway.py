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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
import pywt
import pydub
import time
import wfdb
import sounddevice as sd
from scipy import signal
import preprocessing_lib as pplib
import feature_extraction_lib as ftelib
import file_process_lib as importlib
from sklearn.preprocessing import OneHotEncoder

# import librosa
# import logging
import scipy.io
# import scipy.signal
# import re

import pickle

from scipy.io import wavfile


import copy

# %% Constants
AVERAGE_WINDOW = 3
SIGNAL_IDX = 704
EXTEND_WINDOW = 8

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
pcg_state_predictions = np.array(
    [max_temporal_modelling(prediction) for prediction in pcg_prediction_labels], dtype=object)


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

print(
    f"ULGSE, patient {ecg_df['ID'][SIGNAL_IDX]}, auscultation point {ecg_df['Auscultation_Point'][SIGNAL_IDX]}")

# %% Visualization of both signals
# Originals (For quality re-annotation)
fig, ax = plt.subplots(2, 1, layout='constrained')
fig.suptitle(
    f"ULGSE, patient {ecg_df['ID'][SIGNAL_IDX]}, auscultation point {ecg_df['Auscultation_Point'][SIGNAL_IDX]} Original")

ax[0].set_title('ECG')
ax[0].plot(ecg_df_original['ECG'][SIGNAL_IDX])
ax[0].grid()

ax[1].set_title('PCG')
ax[1].plot(pcg_df_original['PCG'][SIGNAL_IDX])
ax[1].grid()

plt.show()

# Processed Signals
fig = plt.figure(layout='constrained')
fig.suptitle(
    f"ULGSE, patient {ecg_df['ID'][SIGNAL_IDX]}, auscultation point {ecg_df['Auscultation_Point'][SIGNAL_IDX]} Raw -Resampled- Signals")

subfigs = fig.subfigures(2, 1, hspace=0)

top = subfigs[0].subplots(2, 1, sharex=True)
subfigs[0].suptitle('Raw Signals')
top[0].plot(ecg_df['ECG'][SIGNAL_IDX], label='ECG')
top[0].set_xticks([])
top[0].legend(loc=3)
top[0].grid()

top[1].plot(pcg_df['PCG'][SIGNAL_IDX], label='PCG')
top[1].set_xticks([])
top[1].legend(loc=3)
top[1].grid()


bot = subfigs[1].subplots(2, 1, sharex=True)
subfigs[1].suptitle('Delineations')

bot[0].set_title('ECG Delineations')
bot[0].plot((pplib.min_max_norm2(
    ecg_df['ECG'][SIGNAL_IDX])*2 + 1.5), label='ECG raw')
bot[0].plot(pplib.resample_delineation(
    ecg_state_predictions[SIGNAL_IDX], 50, 1000), label='ECG Delineation')
bot[0].set_xticks([])
bot[0].legend(loc=3)
bot[0].grid()

bot[1].set_title('PCG Delineations')
bot[1].plot((pplib.min_max_norm2(
    pcg_df['PCG'][SIGNAL_IDX])*2 + 1.5), label='PCG raw')
bot[1].plot(pplib.resample_delineation(
    pcg_state_predictions[SIGNAL_IDX], 50, 1000), label='PCG Delineation')
bot[1].set_xticks([])
bot[1].legend(loc=3)
bot[1].grid()

fig.show()


# %% Visualize alignments
# fig, ax = plt.subplots(2, 1, sharex=True, layout='constrained')
# fig.suptitle(
#     f"ULGSE, patient {ecg_df['ID'][SIGNAL_IDX]}, auscultation point {ecg_df['Auscultation_Point'][SIGNAL_IDX]} Synchronization Analysis")

# ax[0].set_title('Unsynchronized Sequences')
# ax[0].plot(ecg_state_predictions[SIGNAL_IDX], label='ECG Sequences')
# ax[0].plot(pcg_state_predictions[SIGNAL_IDX], label='PCG Sequences')
# ax[0].legend(loc=3)
# ax[0].grid()

# ax[1].set_title('Synchronized Sequences')
# ax[1].plot(corrected_ecg, label='ECG Sequences')
# ax[1].plot(corrected_pcg, label='PCG Sequences')
# ax[1].legend(loc=3)
# ax[1].grid()

# plt.show()

plt.figure(layout='constrained')
plt.title(
    f"ULGSE, patient {ecg_df['ID'][SIGNAL_IDX]}, auscultation point {ecg_df['Auscultation_Point'][SIGNAL_IDX]} Delineations Overlapped")
plt.plot(ecg_state_predictions[SIGNAL_IDX], label='ECG Sequences')
plt.plot(pcg_state_predictions[SIGNAL_IDX], label='PCG Sequences')
plt.legend(loc=3)
plt.grid()
plt.show()

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

# %% Compute the scores for each segment


ecg_signal = ecg_state_predictions[SIGNAL_IDX]
pcg_signal = pcg_state_predictions[SIGNAL_IDX]
lambda_penalty = 0.1
min_duration = 1
min_overlap = 1

# Extract intervals for physiologically valid segments
ecg_qrs = get_intervals(ecg_signal, target_label=0, min_duration=min_duration)
ecg_twave = get_intervals(ecg_signal, target_label=2,
                          min_duration=min_duration)
pcg_s1 = get_intervals(pcg_signal, target_label=0, min_duration=min_duration)
pcg_s2 = get_intervals(pcg_signal, target_label=2, min_duration=min_duration)

# ECG-to-PCG matching
# Extend to the right ECG segments
qrs_extended = pplib.extend_intervals(ecg_qrs, 'right', EXTEND_WINDOW)
twave_extended = pplib.extend_intervals(ecg_twave, 'right', EXTEND_WINDOW)

# linear combination strategy
match_qrs = match_intervals(qrs_extended, pcg_s1, min_overlap=min_overlap)
match_twave = match_intervals(twave_extended, pcg_s2, min_overlap=min_overlap)

# QRS should match with S1, T-wave should match with S2
total_ecg = len(qrs_extended) + len(twave_extended)
matches_ecg = match_qrs + match_twave
# Apply linear penalty for missing matches
score_ecg_to_pcg = (matches_ecg - lambda_penalty *
                    (total_ecg - matches_ecg)) / total_ecg if total_ecg > 0 else 0

print('Linear combination of metrics E2P: ', score_ecg_to_pcg)

# matching minima strategy
# Weighted Recall
# Compute score for QRS: if no QRS intervals, assume worst match (score = 0)
if len(qrs_extended) > 0:
    score_qrs = (match_qrs - lambda_penalty *
                 (len(qrs_extended) - match_qrs)) / len(qrs_extended)
else:
    score_qrs = 0

# Compute score for T-wave: if no T-wave intervals, assume worst match (score = 0)
if len(twave_extended) > 0:
    score_twave = (match_twave - lambda_penalty *
                   (len(twave_extended) - match_twave)) / len(twave_extended)
else:
    score_twave = 1

# Overall ECG-to-PCG score is the minimum of the two scores to ensure both segments align well
score_ecg_to_pcg = min(score_qrs, score_twave)

print('Minimum-based metrics E2P: ', score_ecg_to_pcg)

# %%
# PCG-to_ECG matching
# Extend PCG segments to the left
s1_extended = pplib.extend_intervals(pcg_s1, 'left', EXTEND_WINDOW)
s2_extended = pplib.extend_intervals(pcg_s2, 'left', EXTEND_WINDOW)

# Linear combination
match_s1 = match_intervals(s1_extended, ecg_qrs, min_overlap=min_overlap)
match_s2 = match_intervals(s2_extended, ecg_twave, min_overlap=min_overlap)


"""Pendiente"""


# QRS should match with S1, T-wave should match with S2
total_ecg = len(ecg_qrs) + len(ecg_twave)
matches_ecg = match_qrs + match_twave
# Apply linear penalty for missing matches
score_ecg_to_pcg = (matches_ecg - lambda_penalty *
                    (total_ecg - matches_ecg)) / total_ecg if total_ecg > 0 else 0

print('Linear combination of metrics E2P: ', score_ecg_to_pcg)
