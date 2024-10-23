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
import file_process_lib as importlib


# Denoising
# Import and analyze the dataset
# Directory containing the files
root_dir = r'..\DatasetCHVNGE'

# Get all pcg files
pcg_files = glob.glob(os.path.join(root_dir, '*.mp3'))

# Extract patient ID and Auscultation Point

# Initialize lists to store file information
patient_ids = []
auscultation_points = []
features_list = []

for file_path in pcg_files:
    # Get the base name of the file
    base_name = os.path.basename(file_path)
    # Remove the extension
    file_name, _ = os.path.splitext(base_name)
    # Split the file name into parts
    parts = file_name.split('_')
    try:
        # Extract patient ID
        patient_id_str = parts[0]
        patient_id = int(patient_id_str)

        # Extract auscultation point (join remaining parts)
        auscultation_point = '_'.join(parts[1:])

        # Load PCG file
        samplerate, pcg = importlib.import_CHVNGE_PCG(file_path)

        # Extract features
        data = np.copy(pcg)
        z_norm = pplib.z_score_standardization(data)

        # Resample 1kHz
        resample = pplib.downsample(z_norm, samplerate, 1000)

        # Schmidt despiking
        despiked_signal = pplib.schmidt_spike_removal(resample, 1000)

        # wavelet denoising
        wavelet_denoised = pplib.wavelet_denoise(
            despiked_signal, 5, wavelet_family='coif4',
            risk_estimator=pplib.val_SURE_threshold,
            shutdown_bands=[-1, 1])

        # Feature Extraction
        # Homomorphic Envelope
        homomorphic = ftelib.homomorphic_envelope(wavelet_denoised, 1000, 50)

        # CWT Scalogram Envelope
        cwt_morl = ftelib.c_wavelet_envelope(wavelet_denoised, 1000, 50,
                                             interest_frequencies=[40, 60])

        cwt_mexh = ftelib.c_wavelet_envelope(
            wavelet_denoised, 1000, 50, wv_family='mexh',
            interest_frequencies=[40, 60])

        # Hilbert Envelope
        hilbert_env = ftelib.hilbert_envelope(wavelet_denoised, 1000, 50)

        # Organize and stack features
        features = np.column_stack(
            (homomorphic, cwt_morl, cwt_mexh, hilbert_env))

        # Append to lists
        patient_ids.append(patient_id)
        auscultation_points.append(auscultation_point)
        features_list.append(features)

    except (ValueError, IndexError):
        print(f"Skipping file with unexpected format: {file_path}")
        continue


# Create the DataFrame
df = pd.DataFrame({
    'Patient_ID': patient_ids,
    'Auscultation_Point': auscultation_points,
    'Features': features_list
})

# Save the DataFrame to a pickle file
output_pickle_path = r'..\preprocessed_CHVNGE_PCG_initial.pkl'
df.to_pickle('output_pickle_path')

print(f"DataFrame saved to {output_pickle_path}")
