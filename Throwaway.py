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

# %% ULSGE Dataset results
# # Import Original
# root_dir = r'..\DatasetCHVNGE\pcg_processed.pkl'
# df = pd.read_pickle(root_dir)
# # Drop empty columns
# df = df.drop(index=[491, 503])
# df = df.reset_index(drop=True)
# # Resample them to 50 Hz
# # Optimize the processing of the dataset
# df['Processed Signal'] = df['Processed Signal'].apply(
#     lambda data: pplib.downsample(data, 1000, 50))

# # Import Predictions
# pred_path = r'..\ULSGE_pred_wv.pkl'
# with open(pred_path, 'rb') as file:
#     predictions = pickle.load(file)

# # Create main figure
# fig = plt.figure(layout='constrained', figsize=(7, 8))
# fig.suptitle('ULSGE Segmentation Results - Wavelet Only')

# subfigs = fig.subfigures(3, 1, hspace=0)

# top = subfigs[0].subplots(2, 1, sharex=True, sharey=True)
# subfigs[0].suptitle('Signal Quality 5')
# top[0].plot(pplib.min_max_norm2(df.iloc[620][2])[380:800])
# top[0].plot(predictions[620][380:800, 0], label='S1')
# top[0].set_xticks([])
# top[0].set_ylim(-1, 1)
# top[0].legend(loc=3)
# top[0].grid()

# top[1].plot(pplib.min_max_norm2(df.iloc[620][2])[380:800])
# top[1].plot(predictions[620][380:800, 2], label='S2')
# top[1].set_xticks([])
# top[1].set_ylim(-1, 1)
# top[1].legend(loc=3)
# top[1].grid()


# mid = subfigs[1].subplots(2, 1, sharex=True, sharey=True)
# subfigs[1].suptitle('Signal Quality 4')
# mid[0].plot(pplib.min_max_norm2(df.iloc[608][2])[380:800])
# mid[0].plot(predictions[608][380:800, 0], label='S1')
# mid[0].set_xticks([])
# mid[0].set_ylim(-1, 1)
# mid[0].legend(loc=3)
# mid[0].grid()

# mid[1].plot(pplib.min_max_norm2(df.iloc[608][2])[380:800])
# mid[1].plot(predictions[608][380:800, 2], label='S2')
# mid[1].set_xticks([])
# mid[1].set_ylim(-1, 1)
# mid[1].legend(loc=3)
# mid[1].grid()


# bot = subfigs[2].subplots(2, 1, sharex=True, sharey=True)
# subfigs[2].suptitle('Signal Quality 3')
# bot[0].plot(pplib.min_max_norm2(df.iloc[601][2])[380:800])
# bot[0].plot(predictions[601][380:800, 0], label='S1')
# bot[0].set_xticks([])
# bot[0].set_ylim(-1, 1)
# bot[0].legend(loc=3)
# bot[0].grid()

# bot[1].plot(pplib.min_max_norm2(df.iloc[601][2])[380:800])
# bot[1].plot(predictions[601][380:800, 2], label='S2')
# bot[1].set_xticks([])
# bot[1].set_ylim(-1, 1)
# bot[1].legend(loc=3)
# bot[1].grid()

# plt.show()
# %%

# pcg_dir = r"..\DatasetCHVNGE\5_AV.mp3"

# # Import audio file
# a = pydub.AudioSegment.from_mp3(pcg_dir)
# samplerate = a.frame_rate
# PCG = np.array(a.get_array_of_samples())
# PCG_bit_width = 16
# PCG_resolution = (2 ** PCG_bit_width)-1
# # Normalize full-scale
# original_data = (PCG) / (PCG_resolution)  # uint 16 bits (scale -0.5;0.5)

# data = np.copy(original_data)
# z_norm = pplib.z_score_standardization(data)

# # Resample 1kHz
# resample = pplib.downsample(z_norm, samplerate, 1000)

# # Schmidt despiking
# despiked_signal = pplib.schmidt_spike_removal(resample, 1000)

# despiked2 = np.copy(despiked_signal)

# # wavelet denoising
# wavelet_denoised = pplib.wavelet_denoise(
#     despiked_signal, 5, wavelet_family='coif4', risk_estimator=pplib.val_SURE_threshold, shutdown_bands=[-1, 1, 2])


# # Butterworth bandpass filtering
# filtered_wavelet = pplib.butterworth_filter(
#     wavelet_denoised, 'bandpass', 4, 1000, [15, 450])
# filtered_just_butt = pplib.butterworth_filter(
#     despiked2, 'bandpass', 4, 1000, [15, 450])
# # %% Plots Pre-processing

# # Subplots
# fig1, axs = plt.subplots(6, figsize=(7, 8))
# fig1.suptitle('ULSGE 5 AV (Quality = 3) Preprocessing Steps')
# axs[0].plot(data)
# axs[0].set_title('Raw Signal')
# axs[0].grid()

# axs[1].plot(z_norm)
# axs[1].set_title('Standardized Signal')
# axs[1].grid()

# axs[2].plot(despiked_signal)
# axs[2].set_title('Schmidt-Despiked Signal')
# axs[2].grid()

# axs[3].plot(wavelet_denoised)
# axs[3].set_title('Only Wavelet Denoised Signal')
# axs[3].grid()

# axs[4].plot(filtered_just_butt)
# axs[4].set_title('Only Butterworth Filtered Signal')
# axs[4].grid()

# axs[5].plot(filtered_wavelet)
# axs[5].set_title('Wavelet Denoised and Butterworth Filtered Signal')
# axs[5].grid()

# for ax in axs:
#     ax.set_xticks([])
#     # ax.set_yticks([])
# fig1.tight_layout()


# # Overlapping
# fig2, ax_ov = plt.subplots(1, figsize=(7, 5.25))
# fig2.suptitle('ULSGE 5 AV (Quality = 3) Preprocessing Strategies Comparison')
# ax_ov.plot(filtered_just_butt[20000:26000],
#            label='Only Butterworth Filtered Signal')
# ax_ov.plot(wavelet_denoised[20000:26000], label='Only Wavelet Denoised Signal')
# ax_ov.plot(filtered_wavelet[20000:26000],
#            label='Wavelet Denoised and Butterworth Filtered Signal')
# ax_ov.grid()
# ax_ov.set_xticks([])
# ax_ov.legend()
# fig2.tight_layout()

# # %% Envelopes
# filtered_pcg = np.copy(filtered_wavelet)
# # Reference PCG
# resample_ref_pcg = pplib.downsample(filtered_pcg, 1000, 50)

# # Homomorphic Envelope
# homomorphic = ftelib.homomorphic_envelope(
#     filtered_pcg, 1000, 50)

# # CWT Scalogram Envelope
# cwt_morl = ftelib.c_wavelet_envelope(filtered_pcg, 1000, 50,
#                                      interest_frequencies=[40, 200])

# cwt_mexh = ftelib.c_wavelet_envelope(
#     filtered_pcg, 1000, 50, wv_family='mexh',
#     interest_frequencies=[40, 200])

# # Hilbert Envelope
# hilbert_env = ftelib.hilbert_envelope(filtered_pcg, 1000, 50)

# # %% Plots Envelopes

# # Subplots
# # Subplots
# fig, axs = plt.subplots(4, figsize=(7, 8))
# fig.suptitle('ULSGE 5 AV (Quality = 3) Envelopes')
# axs[0].plot(homomorphic)
# axs[0].set_title('Homomorphic Envelope')
# axs[0].grid()

# axs[1].plot(cwt_morl)
# axs[1].set_title('Morlet Wavelet Scalogram Envelope')
# axs[1].grid()

# axs[2].plot(cwt_mexh)
# axs[2].set_title('Mexican Hat Wavelet Scalogram Envelope')
# axs[2].grid()

# axs[3].plot(hilbert_env)
# axs[3].set_title('Hilbert Envelope')
# axs[3].grid()

# for ax in axs:
#     ax.set_xticks([])
#     # ax.set_yticks([])
# fig.tight_layout()


# # Overlapping
# fig, ax = plt.subplots(1, figsize=(7, 5.25))
# fig.suptitle('ULSGE 5 AV (Quality = 3) Envelopes Comparison')
# ax.plot(cwt_morl[1000:1300], label='Morlet Wavelet Scalogram Envelope')
# ax.plot(cwt_mexh[1000:1300], label='Mexican Hat Wavelet Scalogram Envelope')
# ax.plot(hilbert_env[1000:1300], label='Hilbert Envelope')
# ax.plot(homomorphic[1000:1300], label='Homomorphic Envelope')
# ax.grid()
# ax.set_xticks([])
# ax.legend()
# fig.tight_layout()

# %% Denoising
# Import and analyze the dataset

# directory = r'../LUDB/data/1'

# # Read as record
# record = wfdb.rdrecord(directory)

# # Read only signals
# # signals, fields = wfdb.rdsamp(directory, channels=[1, 3, 5, 6])
# signals, _ = wfdb.rdsamp(directory)  # <--

# # Read annotations
# ann = wfdb.rdann(directory, extension="i")

# # indices where the annotation is applied
# annotation_index_i = ann.sample

# # symbol order of the annotations
# annotation_vector_i = ann.symbol


# ann = wfdb.rdann(directory, extension="ii")

# # indices where the annotation is applied
# annotation_index_ii = ann.sample

# # symbol order of the annotations
# annotation_vector_ii = ann.symbol

# ann = wfdb.rdann(directory, extension="iii")

# # indices where the annotation is applied
# annotation_index_iii = ann.sample

# # symbol order of the annotations
# annotation_vector_iii = ann.symbol

# %% Process full LUDB dataset

ludb_df = pd.read_pickle(r'..\LUDB\ludb_full.pkl')

# Processing pipeline
ecg_raw = ludb_df.signal[14]

# ECG_path = r"../DatasetCHVNGE/5_TV.raw"
# ECG = np.loadtxt(ECG_path, delimiter=",", dtype=int)
# ECG_bit_width = 12
# ECG_resolution = (2 ** ECG_bit_width)-1
# ecg_raw = ECG / ECG_resolution

fs = 500  # 500 sps original frequency

# Standardization
ecg_zscore = pplib.z_score_standardization(ecg_raw)

# Bandpass filter directly
ecg_filter_bp = pplib.butterworth_filter(
    ecg_zscore, 'bandpass', 6, fs, [0.5, 100])

# Notch filter. Remove 50 Hz band
b_notch, a_notch = signal.iirnotch(50, Q=30, fs=fs)

notched_bandpass = signal.filtfilt(b_notch, a_notch, ecg_filter_bp)
notched_bandpass -= np.median(notched_bandpass)


plt.figure()
plt.title('Pre-processing')
plt.plot(ecg_zscore, label='standardized ecg')
plt.plot(notched_bandpass, label='notched bandpass ecg')
plt.grid()
plt.legend()
plt.show()

# %% features bandpass filter

hilbert_bp = ftelib.hilbert_envelope(notched_bandpass, fs, 50)
shannon_bp = ftelib.shannon_envelopenergy(notched_bandpass, fs, 50)
homomorphic_bp = ftelib.homomorphic_envelope(
    notched_bandpass, fs, 50, median_window=21)
hamming_bp = ftelib.hamming_smooth_envelope(notched_bandpass, 21, fs, 50)

plt.figure()
plt.title('Features from bandpass filters')
plt.plot(hilbert_bp, label='hilbert envelope')
plt.plot(shannon_bp, label='shannon envelope')
plt.plot(homomorphic_bp, label='homomorphic envelope')
plt.plot(hamming_bp, label='smooth energy')
plt.grid()
plt.legend()
plt.show()

# %% Process labels
labels_raw = np.array(ludb_df.label[14])

# Label Processing
desired_order = ['x', 'p', 'N', 't']
# Extract the unique labels and reshape the labels for one-hot encoding
unique_labels = np.unique(labels_raw)
# Ensure that the desired order matches the unique labels
assert set(desired_order) == set(
    unique_labels), "The desired order does not match the unique labels"

# Reshape the labels to a 2D array to fit the OneHotEncoder input
labels_reshaped = labels_raw.reshape(-1, 1)

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False,
                        categories=[desired_order])

# Fit and transform the labels to one-hot encoding
# one_hot_encoded = np.abs(pplib.downsample(
#     encoder.fit_transform(propagated_labels_reshaped), samplerate, 50))

one_hot_encoded = encoder.fit_transform(labels_reshaped)
one_hot_encoded = one_hot_encoded[::10, :]
