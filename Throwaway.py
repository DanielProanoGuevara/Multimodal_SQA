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
import preprocessing_lib as pplib
import feature_extraction_lib as ftelib

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


# %%
# Import and analyze the dataset

# Directory containing the files
directory = '../Physionet_2016_training/training-a/a0288.wav'

# Load .wav files
samplerate, original_data = wavfile.read(directory)

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
# plt.grid()
# plt.legend(loc='lower right')
# plt.show()

# plt.figure()
# Homomorphic envelope
homomorphic = ftelib.homomorphic_envelope(wavelet_denoised, 1000, 50)
plt.plot(homomorphic, label='homomorphic envelope')
plt.grid()
plt.legend(loc='lower right')
plt.show()


# %%
