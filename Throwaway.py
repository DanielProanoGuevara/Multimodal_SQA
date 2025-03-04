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
SIGNAL_IDX = 693

# %% Import PCG
root_dir = r'..\DatasetCHVNGE\pcg_ulsge.pkl'
pcg_df_original = pd.read_pickle(root_dir)
# Deep copy
pcg_df = copy.deepcopy(pcg_df_original)
# Resample them to 50 Hz
pcg_df['PCG'] = pcg_df['PCG'].apply(
    lambda data: pplib.downsample(data, 3000, 50))

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
pcg_pred_labels = [ftelib.reverse_one_hot_encoding(
    pred) for pred in pcg_processed_predictions]
pcg_prediction_labels = copy.deepcopy(pcg_pred_labels)
pcg_state_predictions = np.array(
    [prediction for prediction in pcg_prediction_labels], dtype=object)

# %% Import ECG
root_dir = r'..\DatasetCHVNGE\ecg_ulsge.pkl'
ecg_df_original = pd.read_pickle(root_dir)
# Deep copy
ecg_df = copy.deepcopy(ecg_df_original)
# Resample them to 50 Hz
ecg_df['ECG'] = ecg_df['ECG'].apply(
    lambda data: pplib.downsample(data, 500, 50))
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
    ecg_df['ECG'][SIGNAL_IDX][300:700])*2 + 1.5), label='ECG raw')
bot[0].plot(ecg_state_predictions[SIGNAL_IDX]
            [300:700], label='ECG Delineation')
bot[0].set_xticks([])
bot[0].legend(loc=3)
bot[0].grid()

bot[1].set_title('PCG Delineations')
bot[1].plot((pplib.min_max_norm2(
    pcg_df['PCG'][SIGNAL_IDX][300:700])*2 + 1.5), label='PCG raw')
bot[1].plot(pcg_state_predictions[SIGNAL_IDX]
            [300:700], label='PCG Delineation')
bot[1].set_xticks([])
bot[1].legend(loc=3)
bot[1].grid()

fig.show()

# %% Alignment analysis

# Use EXCLUSSIVELY the neggative lag, being that ECG _always_ precede PCG
e_to_p_correlation = signal.correlate(
    ecg_state_predictions[SIGNAL_IDX], pcg_state_predictions[SIGNAL_IDX], mode='full')
e_to_p_corr_lags = signal.correlation_lags(
    ecg_state_predictions[SIGNAL_IDX].size, pcg_state_predictions[SIGNAL_IDX].size, mode='full')
e_to_p_lag = e_to_p_corr_lags[np.argmax(e_to_p_correlation)]
print('ECG to PCG lag: ', e_to_p_lag)

if np.sign(e_to_p_lag) == 1:
    corrected_pcg = pcg_state_predictions[SIGNAL_IDX][:-abs(e_to_p_lag)]
    corrected_ecg = ecg_state_predictions[SIGNAL_IDX][abs(e_to_p_lag):]
    # This might be signaling inverted labeling for PCG
elif np.sign(e_to_p_lag) == -1:
    corrected_pcg = pcg_state_predictions[SIGNAL_IDX][abs(e_to_p_lag):]
    corrected_ecg = ecg_state_predictions[SIGNAL_IDX][:-abs(e_to_p_lag)]
elif np.sign(e_to_p_lag) == 0:
    pass

# Alignment verification
correlation = signal.correlate(corrected_ecg, corrected_pcg, mode='full')
corr_lags = signal.correlation_lags(
    corrected_ecg.size, corrected_pcg.size, mode='full')
lag = corr_lags[np.argmax(correlation)]
print('ECG to PCG lag after correction: ', lag)

# %% Visualize alignments
fig, ax = plt.subplots(2, 1, sharex=True, layout='constrained')
fig.suptitle(
    f"ULGSE, patient {ecg_df['ID'][SIGNAL_IDX]}, auscultation point {ecg_df['Auscultation_Point'][SIGNAL_IDX]} Synchronization Analysis")

ax[0].set_title('Unsynchronized Sequences')
ax[0].plot(ecg_state_predictions[SIGNAL_IDX][300:700], label='ECG Sequences')
ax[0].plot(pcg_state_predictions[SIGNAL_IDX][300:700], label='PCG Sequences')
ax[0].legend(loc=3)
ax[0].grid()

ax[1].set_title('Synchronized Sequences')
ax[1].plot(corrected_ecg[300:700], label='ECG Sequences')
ax[1].plot(corrected_pcg[300:700], label='PCG Sequences')
ax[1].legend(loc=3)
ax[1].grid()

plt.show()
