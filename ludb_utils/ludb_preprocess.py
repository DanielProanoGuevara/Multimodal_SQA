# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:36:54 2025

@author: danie
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy import signal
import preprocessing_lib as pplib
import feature_extraction_lib as ftelib


ludb_df = pd.read_pickle(r'..\LUDB\ludb_full.pkl')
FS = 500  # 500 sps original frequency

# Notch filter. Remove 50 Hz band
B, A = signal.iirnotch(50, Q=30, fs=FS)

# Label Processing
DESIRED_ORDER = ['x', 'p', 'N', 't']

# Initialize the OneHotEncoder
ENCODER = OneHotEncoder(sparse_output=False,
                        categories=[DESIRED_ORDER])
# Pre-fit to avoid multiple fits
ENCODER.fit(np.array(DESIRED_ORDER).reshape(-1, 1))

# Split files into train and test
split = int(0.8 * len(ludb_df))

train_set = ludb_df.iloc[:split]
validation_set = ludb_df.iloc[split:]


def process_files(dataset):
    # Globals
    global FS, B, A, DESIRED_ORDER, ENCODER

    # Initialize lists to store data
    hilbert_list, shannon_list, homomorphic_list, hamming_list = [], [], [], []
    labels_list, ids_list = [], []

    for i, row in dataset.iterrows():
        try:
            # Signal processing
            ecg_raw = row.get('signal')
            patient_id = row.get('id')
            # Standardization
            ecg_zscore = pplib.z_score_standardization(ecg_raw)
            # Bandpass
            ecg_bandpass = pplib.butterworth_filter(
                ecg_zscore, 'bandpass', order=6, fs=FS, fc=[0.5, 100])

            # Notch. Remove 50 Hz
            ecg_notch = signal.filtfilt(B, A, ecg_bandpass)
            # Detrend
            ecg_notch -= np.median(ecg_notch)

            hilbert_env = ftelib.hilbert_envelope(
                ecg_notch, fs_inicial=FS, fs_final=50)
            shannon_env = ftelib.shannon_envelopenergy(
                ecg_notch, fs_inicial=FS, fs_final=50)
            homomorphic_env = ftelib.homomorphic_envelope(
                ecg_notch, median_window=21, fs_inicial=FS, fs_final=50)
            hamming_env = ftelib.hamming_smooth_envelope(
                ecg_notch, window_size=21, fs_inicial=FS, fs_final=50)

            # Label processing
            labels_raw = np.array(row.get('label', []))

            # Reshape the label vector for processing
            labels_reshaped = labels_raw.reshape(-1, 1)

            # Fit and transform the labels to one-hot encoding
            one_hot_encoded = ENCODER.transform(labels_reshaped)
            # Decimate to match features sampling rate
            one_hot_encoded = one_hot_encoded[::10, :]

            ids_list.append(patient_id)
            hilbert_list.append(hilbert_env)
            shannon_list.append(shannon_env)
            homomorphic_list.append(homomorphic_env)
            hamming_list.append(hamming_env)
            labels_list.append(one_hot_encoded)

        except Exception as e:
            print(f"Error processing patient ID {dataset.id[i]}: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame({
        'Patient ID': ids_list,
        'Hilbert': hilbert_list,
        'Shannon': shannon_list,
        'Homomorphic': homomorphic_list,
        'Hamming': hamming_list,
        'Labels': labels_list,
    })
    return df


train_df = process_files(train_set)
validation_df = process_files(validation_set)

# Save the DataFrames to pickle files
train_pickle_path = r'..\train_ludb.pkl'
validation_pickle_path = r'..\validation_ludb.pkl'

train_df.to_pickle(train_pickle_path)
print(f"Train DataFrame saved to {train_pickle_path}")
validation_df.to_pickle(validation_pickle_path)
print(f"Validate DataFrame saved to {validation_pickle_path}")
