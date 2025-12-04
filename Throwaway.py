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
from scipy import stats
# import scipy.signal
# import re

import pickle

from scipy.io import wavfile

from scipy.stats import pearsonr, kendalltau, spearmanr


import copy


# %% Import ULSGE pickle

class UlsgeAccessor:
    def __init__(self, pkl_file_path):
        self.df = pd.read_pickle(pkl_file_path)

    def _get_index(self, patient_id, auscultation_point):
        match = self.df[(self.df['ID'] == str(patient_id)) &
                        (self.df['Auscultation_Point'] == auscultation_point)]
        if match.empty:
            raise ValueError(
                f"No entry found for ID={patient_id} and Point={auscultation_point}")
        return match.index[0]

    def get_ecg(self, patient_id, auscultation_point):
        idx = self._get_index(patient_id, auscultation_point)
        return self.df.at[idx, 'ECG']

    def get_pcg(self, patient_id, auscultation_point):
        idx = self._get_index(patient_id, auscultation_point)
        return self.df.at[idx, 'PCG']

    def get_ecg_clean(self, patient_id, auscultation_point):
        data = self.get_ecg(patient_id, auscultation_point)
        FS = 500  # 500 sps original frequency
        # Notch filter. Remove 50 Hz band
        B, A = signal.iirnotch(50, Q=30, fs=FS)
        # Butterworth bandpass filtering
        ecg_bandpass = pplib.butterworth_filter(
            data, 'bandpass', order=6, fs=FS, fc=[0.5, 150])
        # Notch. Remove 50 Hz
        ecg_notch = signal.filtfilt(B, A, ecg_bandpass)
        # Detrend
        ecg_notch -= np.median(ecg_notch)
        return ecg_notch

    def get_pcg_clean(self, patient_id, auscultation_point):
        data = self.get_pcg(patient_id, auscultation_point)
        # Schmidt despiking
        despiked_signal = pplib.schmidt_spike_removal(data, 8000)
        # Butterworth bandpass filtering
        filtered_pcg = pplib.butterworth_filter(
            despiked_signal, 'bandpass', 4, 8000, [15, 450])
        return despiked_signal


ulsge = UlsgeAccessor(r'..\DatasetCHVNGE\compiled_dataset.pkl')

# %% Create plots
# patient_id = 1
# paints = ['AV', 'PV', 'TV', 'MV']

# # prepare plot
# fig, axes = plt.subplots(2, 4, figsize=(20, 8))
# fig.suptitle(f'Patient {patient_id}', fontsize=18)

# # Row 1: ECG_AV, ECG_PV, PCG_AV, ECG_PV again
# axes[0, 0].plot(ulsge.get_ecg(patient_id, 'AV'))
# axes[0, 0].set_title('ECG - AV')

# axes[0, 1].plot(ulsge.get_ecg(patient_id, 'PV'))
# axes[0, 1].set_title('ECG - PV')

# axes[0, 2].plot(ulsge.get_pcg(patient_id, 'AV'))
# axes[0, 2].set_title('PCG - AV')

# axes[0, 3].plot(ulsge.get_ecg(patient_id, 'PV'))
# axes[0, 3].set_title('ECG - PV (again)')

# # Row 2: ECG_TV, ECG_MV, PCG_TV, ECG_MV again
# axes[1, 0].plot(ulsge.get_ecg(patient_id, 'TV'))
# axes[1, 0].set_title('ECG - TV')

# axes[1, 1].plot(ulsge.get_ecg(patient_id, 'MV'))
# axes[1, 1].set_title('ECG - MV')

# axes[1, 2].plot(ulsge.get_pcg(patient_id, 'TV'))
# axes[1, 2].set_title('PCG - TV')

# axes[1, 3].plot(ulsge.get_ecg(patient_id, 'MV'))
# axes[1, 3].set_title('ECG - MV (again)')

# # Format
# for ax in axes.flat:
#     ax.set_xlabel('Sample Index')
#     ax.set_ylabel('Amplitude')
#     ax.grid(True)

# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()
id = 11
focus = 'PV'
# plt.figure(figsize=(18, 4))
# # plt.plot(ulsge.get_ecg_clean(id, focus))
# plt.plot(ulsge.get_ecg_clean(id, focus)[0:4500])
# plt.grid(True)

# plt.figure(figsize=(18, 4))
# # plt.plot(ulsge.get_pcg_clean(id, focus))
# plt.plot(ulsge.get_pcg_clean(id, focus)[0:72000])
# plt.grid(True)


# IEEE 1-column figure (~3.5 inches wide)
fig, axs = plt.subplots(2, 1, figsize=(18, 8))

# ECG subplot
axs[0].plot(pplib.min_max_norm(ulsge.get_ecg_clean(
    id, focus)[0:4500]), color='k', linewidth=1.2)
axs[0].grid(True)

# PCG subplot
axs[1].plot(pplib.min_max_norm(ulsge.get_pcg_clean(
    id, focus)[0:72000]), color='k', linewidth=1.2)
axs[1].grid(True)

# Tight layout with extra padding for clarity
plt.tight_layout(pad=2.5)

plt.show()
