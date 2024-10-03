# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:16:31 2024

@author: danie
"""

# %%
# Imports

import os
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
directory_signal = '../Physionet_2016_training/training-a/a0288.wav'
directory_labels = '../Physionet_2016_labels/training-a-Aut/a0288_StateAns0.mat'

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
plt.plot(wav_env_mexh, label='Hilbert envelope')

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
# %%
# Training Patches Creation

X, y = ftelib.create_patches(
    [homomorphic, wav_env_morl, wav_env_mexh, dwt_envelope3, hilbert_env],
    one_hot_encoded,
    50,
    1.5,
    0.5)
