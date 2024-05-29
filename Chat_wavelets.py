# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:20:41 2024

@author: danie
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt

# Function to estimate noise standard deviation using MAD
def estimate_sigma(coeffs):
    return np.median(np.abs(coeffs - np.median(coeffs))) / 0.6745

# Universal Threshold
def universal_threshold(coeffs):
    sigma = estimate_sigma(coeffs)
    return sigma * np.sqrt(2 * np.log(len(coeffs)))

# SURE Threshold
def sure_threshold(coeffs):
    n = len(coeffs)
    sorted_coeffs = np.sort(np.abs(coeffs))
    risks = np.zeros(n)
    for i in range(n):
        t = sorted_coeffs[i]
        risks[i] = (n - 2 * (i + 1) + np.sum(np.minimum(coeffs ** 2, t ** 2)))
    optimal_idx = np.argmin(risks)
    return sorted_coeffs[optimal_idx]

# Minimax Threshold (using approximation for large N)
def minimax_threshold(coeffs):
    n = len(coeffs)
    return 0.3936 + 0.1829 * np.log(n)

# Apply thresholding function
def apply_threshold(coeffs, threshold, method='soft'):
    if method == 'soft':
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)
    elif method == 'hard':
        return coeffs * (np.abs(coeffs) >= threshold)

# Simulated ECG signal (for demonstration purposes)
fs = 500  # Sampling frequency in Hz
t = np.arange(0, 10, 1/fs)  # Time vector
# Synthetic ECG signal (combination of sinusoids) + noise
ecg = np.sin(2 * np.pi * 1.7 * t) + 0.5 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(len(t)) + 5

# Perform wavelet decomposition
wavelet = 'db4'
coeffs = pywt.wavedec(ecg, wavelet, level=5)

# Threshold each level of coefficients using different methods
threshold_universal = universal_threshold(coeffs[-1])
threshold_sure = sure_threshold(coeffs[-1])
threshold_minimax = minimax_threshold(coeffs[-1])

# Apply soft thresholding
coeffs_universal = [apply_threshold(c, threshold_universal) if i > 0 else c for i, c in enumerate(coeffs)]
coeffs_sure = [apply_threshold(c, threshold_sure) if i > 0 else c for i, c in enumerate(coeffs)]
coeffs_minimax = [apply_threshold(c, threshold_minimax) if i > 0 else c for i, c in enumerate(coeffs)]

# Reconstruct the signal from the thresholded coefficients
denoised_ecg_universal = pywt.waverec(coeffs_universal, wavelet)
denoised_ecg_sure = pywt.waverec(coeffs_sure, wavelet)
denoised_ecg_minimax = pywt.waverec(coeffs_minimax, wavelet)

# Plot the original and denoised ECG signals
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(t, ecg, label='Noisy ECG')
plt.title('Noisy ECG Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t, denoised_ecg_universal, label='Denoised ECG (Universal Threshold)', color='orange')
plt.title('Denoised ECG Signal using Universal Threshold')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t, denoised_ecg_sure, label='Denoised ECG (SURE Threshold)', color='green')
plt.title('Denoised ECG Signal using SURE Threshold')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t, denoised_ecg_minimax, label='Denoised ECG (Minimax Threshold)', color='red')
plt.title('Denoised ECG Signal using Minimax Threshold')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
