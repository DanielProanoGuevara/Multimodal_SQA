# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:32:40 2024

@author: danie
"""
# Libraries import
import os
import glob
import numpy as np
import pandas as pd
import preprocessing_lib as pplib
import feature_extraction_lib as ftelib


def process_signals_from_pkl(file_path):
    """
    Process signals from a pickle file containing a DataFrame with processed signals.

    Parameters:
        file_path (str): Path to the pickle file.

    Returns:
        pd.DataFrame: DataFrame with extracted features for each processed signal.
    """
    # Load the DataFrame from the pickle file
    df = pd.read_pickle(file_path)

    processed_features = []

    for _, row in df.iterrows():
        try:
            data = row['Processed Signal']

            # Schmidt despiking
            despiked_signal = pplib.schmidt_spike_removal(data, 1000)

            # Wavelet denoising
            wavelet_denoised = pplib.wavelet_denoise(
                despiked_signal, 5, wavelet_family='coif4',
                risk_estimator=pplib.val_SURE_threshold,
                shutdown_bands=[-1, 1, 2]
            )

            # Butterworth bandpass filtering
            filtered_pcg = pplib.butterworth_filter(
                wavelet_denoised, 'bandpass', 4, 1000, [15, 450])

            # Feature extraction
            # Homomorphic Envelope
            homomorphic = ftelib.homomorphic_envelope(filtered_pcg, 1000, 50)

            # CWT Scalogram Envelope
            cwt_morl = ftelib.c_wavelet_envelope(
                filtered_pcg, 1000, 50, interest_frequencies=[40, 200]
            )

            cwt_mexh = ftelib.c_wavelet_envelope(
                filtered_pcg, 1000, 50, wv_family='mexh',
                interest_frequencies=[40, 200]
            )

            # Hilbert Envelope
            hilbert_env = ftelib.hilbert_envelope(filtered_pcg, 1000, 50)

            # Organize and stack features
            features = np.column_stack((
                homomorphic, cwt_morl, cwt_mexh, hilbert_env
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
root_dir = r'..\DatasetCHVNGE\pcg_processed.pkl'
features_df = process_signals_from_pkl(root_dir)

# Save or inspect the resulting DataFrame
features_df.to_pickle(r'..\features_signals.pkl')
print(features_df)
