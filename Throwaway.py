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


# %%
# Import and analyze the dataset

# Directory containing the files
directory = '../Physionet_2016_training/training-a/a0001.wav'

# Load and combine .wav files
samplerate, data = wavfile.read(directory)

# Print the shape of the combined array
print(data.shape)

# %%
# Denoise

wavelet = 'coif4'


# Plot the original, noisy, and denoised signals
plt.figure(figsize=(10, 6))
plt.plot(data, label='Noisy Signal')
plt.plot(denoised_signal, label='Denoised Signal')
plt.legend()
plt.show()
