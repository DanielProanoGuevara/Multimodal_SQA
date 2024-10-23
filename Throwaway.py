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
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.stats import norm
import pywt
import pydub
import time
import sounddevice as sd
from scipy import signal
from scipy.io.wavfile import write
import preprocessing_lib as pplib
import feature_extraction_lib as ftelib
import file_process_lib as importlib

# %%
# Throwaway Functions


def load_wav_files(directory):
    # List to store the data from all .wav files
    data_list = []

    # Loop through all files in the given directory
    for filename in os.listdir(directory):
        # Check if the file is a .wav file
        if filename.endswith('.wav'):
            # Get the full path to the file
            file_path = os.path.join(directory, filename)
            # Read the .wav file
            samplerate, data = wavfile.read(file_path)
            # Append the data to the list
            data_list.append(data)

    # Combine all data into a single numpy array
    combined_data = np.concatenate(data_list)

    return combined_data


def plot_histogram_with_gaussian(data):
    # Calculate histogram
    counts, bins, _ = plt.hist(
        data, bins=50, density=True, alpha=0.6, color='g')

    # Fit a Gaussian distribution to the data
    mu, std = norm.fit(data)

    # Plot the Gaussian fit
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)

    title = f"Fit results: mu = {mu:.2f},  std = {std:.2f}"
    plt.title(title)

    plt.show()

    return mu, std


def ValSUREThresh(X):
    """
    Adaptive Threshold Selection Using Principle of SURE.

    Parameters
    ----------
    X: array
         Noisy Data with Std. Deviation = 1

    Returns
    -------
    tresh: float
         Value of Threshold

    """
    n = np.size(X)

    # a = mtlb_sort(abs(X)).^2
    a = np.sort(np.abs(X))**2

    c = np.linspace(n-1, 0, n)
    s = np.cumsum(a)+c*a
    risk = (n - (2 * np.arange(n)) + s)/n
    # risk = (n-(2*(1:n))+(cumsum(a,'m')+c(:).*a))/n;
    ibest = np.argmin(risk)
    THR = np.sqrt(a[ibest])
    return THR


def nmse(x, y):
    return (np.mean((x - y) ** 2) / np.mean(x ** 2))


# %% Denoising
# Import and analyze the dataset
# Directory containing the files
root_dir = r'..\DatasetCHVNGE'

# Get all pcg files
pcg_files = glob.glob(os.path.join(root_dir, '*.mp3'))

# %% Extract patient ID and Auscultation Point

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
