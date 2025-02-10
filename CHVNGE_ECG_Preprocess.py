# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:33:03 2025

@author: danie
"""


# Libraries import
import os
import glob
import numpy as np
import pandas as pd
from scipy import signal
import preprocessing_lib as pplib
import feature_extraction_lib as ftelib

FS = 500  # 500 sps original frequency

# Notch filter. Remove 50 Hz band
B, A = signal.iirnotch(50, Q=30, fs=FS)


def process_signals_from_pkl(file_path):
    """
    Process signals from a pickle file containing a DataFrame with processed signals.

    Parameters:
        file_path (str): Path to the pickle file.

    Returns:
        pd.DataFrame: DataFrame with extracted features for each processed signal.
    """
    # Globals
    global FS, B, A

    # Load the DataFrame from the pickle file
    df = pd.read_pickle(file_path)

    processed_features = []

    for _, row in df.iterrows():
        try:
            data = row['Signal']

            # Butterworth bandpass filtering
            ecg_bandpass = pplib.butterworth_filter(
                data, 'bandpass', order=6, fs=FS, fc=[0.5, 100])

            # Notch. Remove 50 Hz
            ecg_notch = signal.filtfilt(B, A, ecg_bandpass)
            # Detrend
            ecg_notch -= np.median(ecg_notch)

            # Feature extraction
            # Hilbert Envelope
            hilbert_env = ftelib.hilbert_envelope(
                ecg_notch, fs_inicial=FS, fs_final=50)
            # Shannon Envelope
            shannon_env = ftelib.shannon_envelopenergy(
                ecg_notch, fs_inicial=FS, fs_final=50)
            # Homomorphic Envelope
            homomorphic_env = ftelib.homomorphic_envelope(
                ecg_notch, median_window=21, fs_inicial=FS, fs_final=50)
            # Smoothing Envelope
            hamming_env = ftelib.hamming_smooth_envelope(
                ecg_notch, window_size=21, fs_inicial=FS, fs_final=50)

            # Organize and stack features
            features = np.column_stack((
                hilbert_env, shannon_env, homomorphic_env, hamming_env
            ))

            # Append features to the list
            processed_features.append({
                'ID': row['ID'],
                'Auscultation Point': row['Auscultation Point'],
                'Features': features
            })
        except Exception as e:
            print(f"Error proccessing Patient ID {row['ID']}: {e}")
            continue  # Skip the file
    # Create a new DataFrame with features
    features_df = pd.DataFrame(processed_features)
    return features_df


# Directory containing the files
root_dir = r'..\DatasetCHVNGE\ecg_ulsge.pkl'
features_df = process_signals_from_pkl(root_dir)

# Save or inspect the resulting DataFrame
features_df.to_pickle(r'..\ulsge_ecg_features.pkl')
