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
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.stats import norm
import pywt
from sklearn.preprocessing import OneHotEncoder
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
directory_signal = r'../Physionet_2016_training/training-e/e00603.wav'
directory_labels = r'../Physionet_2016_labels/training-e-Aut/e00603_StateAns0.mat'

# # Load .wav files
# samplerate, original_data = wavfile.read(directory)

samplerate, original_data, propagated_labels = importlib.import_physionet_2016(
    directory_signal, directory_labels)

# original_data = pplib.resolution_normalization(original_data, 15)
plt.figure()

# create copy of original data
data = np.copy(original_data)
plt.plot(data, label='original')

# standardize data
z_norm = pplib.z_score_standardization(data)
plt.plot(z_norm, label='standardized')
plt.grid()
plt.legend(loc='lower right')
plt.show()

# resample 1k Hz
plt.figure()
resample = pplib.downsample(z_norm, samplerate, 1000)
plt.plot(resample, label='resampled 1kHz')

# Schmidt despiking
despiked_signal = pplib.schmidt_spike_removal(resample, 1000)
plt.plot(despiked_signal, label='despiked signal')

# wavelet denoising
wavelet_denoised = pplib.wavelet_denoise(despiked_signal, 5, wavelet_family='coif4',
                                         risk_estimator=pplib.val_SURE_threshold,
                                         shutdown_bands=[-1])
plt.plot(wavelet_denoised, label='wavelet denoised')
plt.grid()
plt.legend(loc='lower right')
plt.show()

# %% Feature Extraction

# Homomorphic envelope
plt.figure()
homomorphic = ftelib.homomorphic_envelope(wavelet_denoised, 1000, 50)
plt.plot(homomorphic, label='homomorphic envelope')


# Wavelet envelope
wav_env_morl = ftelib.c_wavelet_envelope(wavelet_denoised, 1000, 50)
plt.plot(wav_env_morl, label='morlet wavelet envelope')

wav_env_mexh = ftelib.c_wavelet_envelope(wavelet_denoised, 1000, 50,
                                         wv_family='mexh')
plt.plot(wav_env_mexh, label='mexican hat wavelet envelope')

dwt_envelope3 = ftelib.d_wavelet_envelope(wavelet_denoised, 1000, 50)
plt.plot(dwt_envelope3, label='DWT envelope lv3')

hilbert_env = ftelib.hilbert_envelope(wavelet_denoised, 1000, 50)
plt.plot(hilbert_env, label='Hilbert envelope')

plt.grid()
plt.legend(loc='lower right')
plt.show()

# %% Label encoding
# Extract the unique labels and reshape the labels for one-hot encoding
unique_labels = np.unique(propagated_labels)

# Reshape the labels to a 2D array to fit the OneHotEncoder input
propagated_labels_reshaped = propagated_labels.reshape(-1, 1)

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, categories=[unique_labels])

# Fit and transform the labels to one-hot encoding
one_hot_encoded = np.abs(pplib.downsample(encoder.fit_transform(
    propagated_labels_reshaped), samplerate, 50))


plt.figure()
plt.plot(pplib.min_max_norm2(wavelet_denoised))
# %%
# # Training Patches Creation

# X, y = ftelib.create_patches(
#     Features=[homomorphic, wav_env_morl,
#               wav_env_mexh, dwt_envelope3, hilbert_env],
#     Labels=one_hot_encoded,
#     Patch_Size=8,
#     Stride=3)

# %%
# # Process all database

# # Path to directories
# wav_dir = r"..\Physionet_2016_training"
# mat_dir = r"..\Physionet_2016_labels"

# # Collect and pair .wav and .mat Files
# # Get all .wav files
# wav_files = glob.glob(os.path.join(wav_dir, '**', '*.wav'), recursive=True)

# # Get all .mat files
# mat_files = glob.glob(os.path.join(mat_dir, '**', '*.mat'), recursive=True)

# # Create Dictionaries to Map Patient ID's to File Paths

# wav_dict = {}
# for wav_path in wav_files:
#     # Extract patient ID (e.g., 'a0001')
#     patient_id = os.path.splitext(os.path.basename(wav_path))[0]
#     wav_dict[patient_id] = wav_path

# mat_dict = {}
# for mat_path in mat_files:
#     # Extract patient ID by removing suffix after '_State' and extension
#     basename = os.path.basename(mat_path)
#     patient_id = basename.split('_State')[0]
#     mat_dict[patient_id] = mat_path

# # Pair the files
# # Find common patient IDs
# common_patient_ids = set(wav_dict.keys()).intersection(set(mat_dict.keys()))

# # Create a list of (patient_id, wav_path, mat_path) tuples
# paired_files = [(patient_id, wav_dict[patient_id], mat_dict[patient_id])
#                 for patient_id in common_patient_ids]

# print(f"Found {len(paired_files)} paired files.")

# # Define the minimal duration in seconds
# MIN_DURATION = 2.0

# # Initialize lists to store data
# patient_ids = []
# features_list = []
# labels_list = []

# # Process all files
# for idx, (patient_id, wav_path, mat_path) in enumerate(paired_files):
#     try:
#         samplerate, original_data, propagated_labels = importlib.import_physionet_2016(
#             wav_path, mat_path)
#         time = original_data.size / samplerate
#         if time < MIN_DURATION:
#             # print(f"Skipping Patient ID {patient_id}: Audio duration {
#             #     time:.2f}s is less than the minimum required {MIN_DURATION}s.")
#             continue  # Skip the file

#         # Process
#         data = np.copy(original_data)
#         z_norm = pplib.z_score_standardization(data)
#         # Resample 1kHz
#         resample = pplib.downsample(z_norm, samplerate, 1000)
#         # Schmidt despiking
#         despiked_signal = pplib.schmidt_spike_removal(resample, 1000)
#         # wavelet denoising
#         wavelet_denoised = pplib.wavelet_denoise(
#             despiked_signal, 5, wavelet_family='coif4', risk_estimator=pplib.val_SURE_threshold, shutdown_bands=[-1])
#         # Feature Extraction
#         # Homomorphic Envelope
#         homomorphic = ftelib.homomorphic_envelope(wavelet_denoised, 1000, 50)
#         # CWT Scalogram Envelope
#         cwt_morl = ftelib.c_wavelet_envelope(wavelet_denoised, 1000, 50)
#         cwt_mexh = ftelib.c_wavelet_envelope(
#             wavelet_denoised, 1000, 50, wv_family='mexh')
#         # 3rd decomposition DWT
#         dwt = ftelib.d_wavelet_envelope(wavelet_denoised, 1000, 50)
#         # Hilbert Envelope
#         hilbert_env = ftelib.hilbert_envelope(wavelet_denoised, 1000, 50)

#         # Label Processing
#         # Extract the unique labels and reshape the labels for one-hot encoding
#         unique_labels = np.unique(propagated_labels)

#         # Reshape the labels to a 2D array to fit the OneHotEncoder input
#         propagated_labels_reshaped = propagated_labels.reshape(-1, 1)

#         # Initialize the OneHotEncoder
#         encoder = OneHotEncoder(sparse_output=False,
#                                 categories=[unique_labels])

#         # Fit and transform the labels to one-hot encoding
#         one_hot_encoded = np.abs(pplib.downsample(
#             encoder.fit_transform(propagated_labels_reshaped), samplerate, 50))

#         # Organize
#         features = [homomorphic, cwt_morl, cwt_mexh, dwt, hilbert_env]
#         labels = one_hot_encoded

#         # Append data to lists
#         patient_ids.append(patient_id)
#         features_list.append(features)
#         labels_list.append(labels)

#     except Exception as e:
#         # print(f"Error proccessing Patient ID {patient_id}: {e}")
#         continue  # Skip the file
