# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:16:31 2024

@author: danie
"""

# %%
# Imports

from scipy import signal
import sounddevice as sd
import wfdb
import time
import pydub
import pywt
from scipy.stats import norm
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import preprocessing_lib as pplib
import feature_extraction_lib as ftelib
import file_process_lib as importlib
import os
import sys

import copy
from scipy.stats import pearsonr, kendalltau, spearmanr
from scipy.io import wavfile
import pickle
from scipy import stats
import scipy.io
from sklearn.preprocessing import OneHotEncoder

# Get the absolute path of the mother folder
origi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add origi folder to sys.path
sys.path.append(origi_path)


# import librosa
# import logging
# import scipy.signal
# import re


# %% Import Class Processor

class UlsgeAccessor:
    def __init__(self, pkl_file_path):
        # Load and store the full original DataFrame
        raw_df = pd.read_pickle(pkl_file_path)

        # Check for required columns
        required_cols = ['ID', 'Auscultation_Point', 'Source', 'ECG', 'PCG']
        for col in required_cols:
            if col not in raw_df.columns:
                raise ValueError(f"Missing required column: {col}")

        self.df_ecg = self._process_ecg(raw_df)
        self.df_pcg = self._process_pcg(raw_df)

    def _process_ecg(self, df):
        df_ecg = df[['ID', 'Auscultation_Point', 'ECG']].copy()
        df_ecg['ECG'] = df_ecg['ECG'].apply(pplib.z_score_standardization)
        return df_ecg

    def _process_pcg(self, df):
        df_pcg = df[['ID', 'Auscultation_Point', 'PCG', 'Source']].copy()

        def process(row):
            try:
                signal = row['PCG']
                if row['Source'] == "Rijuven":
                    signal = pplib.downsample(signal, 8000, 3000)
                return pplib.z_score_standardization(signal)
            except Exception:
                return None

        df_pcg['PCG'] = df_pcg.apply(process, axis=1)
        df_pcg = df_pcg[df_pcg['PCG'].notnull()].drop(columns=['Source'])
        return df_pcg

    def _get_index(self, df, patient_id, auscultation_point):
        match = df[(df['ID'] == str(patient_id)) &
                   (df['Auscultation_Point'] == auscultation_point)]
        if match.empty:
            raise ValueError(
                f"No entry found for ID={patient_id} and Point={auscultation_point}")
        return match.index[0]

    def get_ecg(self, patient_id, auscultation_point):
        idx = self._get_index(self.df_ecg, patient_id, auscultation_point)
        return self.df_ecg.at[idx, 'ECG']

    def get_pcg(self, patient_id, auscultation_point):
        idx = self._get_index(self.df_pcg, patient_id, auscultation_point)
        return self.df_pcg.at[idx, 'PCG']

    def get_ecg_features(self, patient_id, auscultation_point):
        FS = 500  # Sampling frequency
        data = self.get_ecg(patient_id, auscultation_point)

        # Notch filter design
        B, A = signal.iirnotch(50, Q=30, fs=FS)

        # Butterworth bandpass filter
        ecg_bandpass = pplib.butterworth_filter(
            data, 'bandpass', order=6, fs=FS, fc=[0.5, 150])

        # Apply notch filter
        ecg_notch = signal.filtfilt(B, A, ecg_bandpass)

        # Detrend by median
        ecg_notch -= np.median(ecg_notch)

        # Feature extraction
        hilbert_env = ftelib.hilbert_envelope(
            ecg_notch, fs_inicial=FS, fs_final=50)
        shannon_env = ftelib.shannon_envelopenergy(
            ecg_notch, fs_inicial=FS, fs_final=50)
        homomorphic_env = ftelib.homomorphic_envelope(
            ecg_notch, median_window=21, fs_inicial=FS, fs_final=50)
        hamming_env = ftelib.hamming_smooth_envelope(
            ecg_notch, window_size=21, fs_inicial=FS, fs_final=50)

        # Stack into feature matrix
        features = np.column_stack(
            (hilbert_env, shannon_env, homomorphic_env, hamming_env))
        return features

    def get_pcg_features(self, patient_id, auscultation_point):
        FS_ORIG = 3000
        FS_FINAL = 1000
        data = self.get_pcg(patient_id, auscultation_point)

        # Downsample
        data_ds = pplib.downsample(data, FS_ORIG, FS_FINAL)

        # Despike
        despiked_signal = pplib.schmidt_spike_removal(data_ds, FS_FINAL)

        # Optional: wavelet denoising
        # wavelet_denoised = pplib.wavelet_denoise(
        #     despiked_signal, 5, wavelet_family='coif4',
        #     risk_estimator=pplib.val_SURE_threshold,
        #     shutdown_bands=[-1, 1, 2]
        # )

        # Butterworth bandpass filter
        filtered_pcg = pplib.butterworth_filter(
            despiked_signal, 'bandpass', order=4, fs=FS_FINAL, fc=[15, 450])

        # Feature extraction
        homomorphic = ftelib.homomorphic_envelope(filtered_pcg, FS_FINAL, 50)
        cwt_morl = ftelib.c_wavelet_envelope(
            filtered_pcg, FS_FINAL, 50, interest_frequencies=[40, 200])
        cwt_mexh = ftelib.c_wavelet_envelope(
            filtered_pcg, FS_FINAL, 50, wv_family='mexh', interest_frequencies=[40, 200])
        hilbert_env = ftelib.hilbert_envelope(filtered_pcg, FS_FINAL, 50)

        # Stack into feature matrix
        features = np.column_stack(
            (homomorphic, cwt_morl, cwt_mexh, hilbert_env))
        return features


################################################################
ulsge = UlsgeAccessor(
    r"C:\Users\danie\Dropbox\PhD\SignalQuality\DatasetCHVNGE\compiled_dataset.pkl")

# %% Create plots
patient_id = 6

# sampling frequencies (adjust these to your actual values)
fs_ecg = 500     # samples per second for ECG
fs_pcg = 3000      # samples per second for PCG

# prepare plot
fig, axes = plt.subplots(4, 2, figsize=(12, 6))
fig.suptitle(f'Patient {patient_id}', fontsize=14)

# helper to plot with time axis


def plot_with_time(ax, signal, fs, title, color=None):
    t = np.arange(len(signal)) / fs
    ax.plot(t, signal, color=color)
    ax.set_title(title)


# Row-wise plotting
plot_with_time(axes[0, 0], ulsge.get_ecg(
    patient_id, 'AV')[:3000], fs_ecg, 'ECG – AV')
plot_with_time(axes[0, 1], ulsge.get_ecg(
    patient_id, 'PV')[:3000], fs_ecg, 'ECG – PV')

plot_with_time(axes[1, 0], ulsge.get_pcg(
    patient_id, 'AV')[:18000], fs_pcg, 'PCG – AV', 'tab:orange')
plot_with_time(axes[1, 1], ulsge.get_pcg(
    patient_id, 'PV')[:18000], fs_pcg, 'PCG – PV', 'tab:orange')

plot_with_time(axes[2, 0], ulsge.get_ecg(
    patient_id, 'TV')[:3000], fs_ecg, 'ECG – TV')
plot_with_time(axes[2, 1], ulsge.get_ecg(
    patient_id, 'MV')[:3000], fs_ecg, 'ECG – MV')

plot_with_time(axes[3, 0], ulsge.get_pcg(
    patient_id, 'TV')[:18000], fs_pcg, 'PCG – TV', 'tab:orange')
plot_with_time(axes[3, 1], ulsge.get_pcg(
    patient_id, 'MV')[:18000], fs_pcg, 'PCG – MV', 'tab:orange')

# Format ticks: only bottom row x-labels, only left column y-labels
for i, row in enumerate(axes):
    for j, ax in enumerate(row):
        # x ticks only on bottom row
        show_x = (i == axes.shape[0] - 1)
        ax.tick_params(axis='x', labelbottom=show_x)
        if show_x:
            ax.set_xlabel('Time (s)')

        # y ticks only on left column
        show_y = (j == 0)
        ax.tick_params(axis='y', labelleft=show_y)
        if show_y:
            ax.set_ylabel('Amplitude')

        ax.grid(True)

plt.tight_layout()

# Save to PDF at high resolution
plt.savefig('auscultation_points_all.pdf', format='pdf', dpi=900)
plt.show()
# %% Quality examples

# Define ECG and PCG (ID, Point) pairs for each quality label
ecg_pairs = [
    ('Q0', 161, 'AV'),
    ('Q1', 145, 'MV'),
    ('Q2', 171, 'PV'),
    ('Q3', 169, 'MV'),
    ('Q4', 159, 'PV'),
    ('Q5', 173, 'TV'),
]

pcg_pairs = [
    ('Q0', 174, 'MV'),
    ('Q1', 168, 'PV'),
    ('Q2', 157, 'MV'),
    ('Q3', 162, 'AV'),
    ('Q4', 164, 'TV'),
    ('Q5', 173, 'PV'),
]

n_rows = len(ecg_pairs)
fig, axes = plt.subplots(n_rows, 2, figsize=(12, 1 * n_rows), sharex=False)
fig.subplots_adjust(hspace=0.5)
fig.suptitle('Signals by Quality and Modality', fontsize=14)

# Set column headers
axes[0, 0].set_title('ECG', fontsize=12)
axes[0, 1].set_title('PCG', fontsize=12)

# Plot each row
for i in range(n_rows):
    # --- ECG plot ---
    label_ecg, pid_ecg, point_ecg = ecg_pairs[i]
    ecg = ulsge.get_ecg(pid_ecg, point_ecg)
    t_ecg = np.arange(len(ecg)) / fs_ecg
    ax_ecg = axes[i, 0]
    ax_ecg.plot(t_ecg, ecg)
    ax_ecg.set_ylabel(label_ecg, rotation=0, labelpad=25, fontsize=12)
    ax_ecg.grid(True)

    # --- PCG plot ---
    label_pcg, pid_pcg, point_pcg = pcg_pairs[i]
    pcg = ulsge.get_pcg(pid_pcg, point_pcg)
    t_pcg = np.arange(len(pcg)) / fs_pcg
    ax_pcg = axes[i, 1]
    ax_pcg.plot(t_pcg, pcg, color='tab:orange')
    ax_pcg.grid(True)

    # Tick formatting
    # x‐ticks only on last row
    show_x = (i == n_rows - 1)
    for ax in (ax_ecg, ax_pcg):
        ax.tick_params(axis='x', labelbottom=show_x)
        if show_x:
            ax.set_xlabel('Time (s)')

        # y‐ticks on right only
        ax.yaxis.set_ticks_position('right')
        ax.tick_params(axis='y', labelleft=False, labelright=True)

plt.tight_layout()
plt.savefig('quality_all_modalities.pdf', format='pdf', dpi=900)
plt.show()

# %% ECG Features

# Inputs
patient_id = 16  # replace with actual ID
auscultation_point = 'TV'  # replace with actual point

# Get original signal and features
ecg_signal = ulsge.get_ecg(patient_id, auscultation_point)
features = ulsge.get_ecg_features(patient_id, auscultation_point)

# Create plot
fig, axes = plt.subplots(5, 1, figsize=(12, 8))
fig.suptitle(
    f"ECG Signal and Features — ID: {patient_id}, Point: {auscultation_point}", fontsize=16)

# Titles for each subplot
titles = ['Original ECG', 'Hilbert Envelope', 'Shannon Energy',
          'Homomorphic Envelope', 'Hamming Smoothed Envelope']

# Plot original
axes[0].plot(ecg_signal)
axes[0].set_title(titles[0])

# Plot features
for i in range(4):
    axes[i + 1].plot(features[:, i])
    axes[i + 1].set_title(titles[i + 1])

# Formatting: hide ticks, keep grid
for ax in axes:
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='y', labelleft=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(True)

# plt.tight_layout()
# Save to PDF at high resolution
plt.savefig('ecg_features.pdf', format='pdf', dpi=900)
plt.show()

# %% PCG Features

# Inputs
patient_id = 16  # replace with actual ID
auscultation_point = 'TV'  # replace with actual point

# Get original signal and features
pcg_signal = ulsge.get_pcg(patient_id, auscultation_point)
features = ulsge.get_pcg_features(patient_id, auscultation_point)

# Create plot
fig, axes = plt.subplots(5, 1, figsize=(12, 8))
fig.suptitle(
    f"PCG Signal and Features — ID: {patient_id}, Point: {auscultation_point}", fontsize=16)

# Titles for each subplot
titles = ['Original PCG', 'Homomorphic Envelope',
          'Morlet Scalogram', 'Mexican Hat Scalogram', 'Hilbert Envelope']

# Plot original
axes[0].plot(pcg_signal)
axes[0].set_title(titles[0])

# Plot features
for i in range(4):
    axes[i + 1].plot(features[:, i])
    axes[i + 1].set_title(titles[i + 1])

# Formatting: hide ticks, keep grid
for ax in axes:
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='y', labelleft=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(True)

plt.tight_layout()
# Save to PDF at high resolution
plt.savefig('pcg_features.pdf', format='pdf', dpi=900)
plt.show()
