"""
ECG and PCG preprocessing library.

@file: preprocessing_lib.py

@coding: utf_8

@description: This module intends to contain all the preprocessing functions
necessary for the digital signal processing of several databases, considering
the ECG and PCG signals. The databases that this code is meant for are:

PCG:
    - Physionet 2016 a-f

@functions:
- resolution_normalization(data, resolution, centered_on_zero): normalizes data
based on the sampling resolution. Takes into consideration if the input data
is already centered at zero, or it needs to be centered.

- z_score_standardization(data): standardizes data based on the z-score. It
makes the data have 0 mean and a standard deviation of 1.

- downsample(input_signal, orig_freq, target_freq): downsamples the input
signal, enforcing that the Shannon-Nyquist criteria is always met by using
polyphase antialias filters. It also manages the interpolation - decimation
opperations needed for meeting the target frequency.

-


@author: Daniel Proaño-Guevara.

@creationDate: 2024-07-12

@version: 0.1
"""

# TODO: Butterworth Filter.
# TODO: Schmidt interference removal.
# TODO: Wavelet denoising. Select bands to remove, select band to infer noise.
# TODO: Homomorphic Envelogram.
# TODO: Hilbert Envelope.
# TODO: Wavelet Envelope.
# TODO: Power Spectral Density Envelope.

import numpy as np
from scipy.signal import firwin, resample_poly


def resolution_normalization(data, resolution, centered_on_zero=True):
    """
    Normalize the data based on the known acquisition resolution.

    Parameters
    ----------
    data : TYPE: int, float, numpy array
        DESCRIPTION: Signal array to be normalized.
    resolution : TYPE: int
        DESCRIPTION: Bit depth in binary. Comes from the knowledge of the
        sampling resolution of the signal.
    centered_on_zero : TYPE: Boolean, optional
        DESCRIPTION: Enables centering the normalized value around zero. The
        default is True.

    Returns
    -------
    normalized_data : TYPE: int, float, numpy array
        DESCRIPTION: data normalized between 0.5 and 0.5 based on the
        acquisition resolution.

    """
    resolution_width = (2**resolution)-1
    if centered_on_zero is True:
        normalized_data = data / resolution_width
    else:
        normalized_data = (data - resolution_width / 2) / (resolution_width)
    return normalized_data


def z_score_standardization(data):
    """
    Standardize the data.

    Parameters
    ----------
    data : TYPE: numpy.ndarray
        DESCRIPTION: The input signal to be standardized.

    Returns
    -------
    TYPE: numpy.ndarray
        DESCRIPTION: The signal standardized.

    """
    mu = np.mean(data)
    std_dev = np.std(data)
    return (data - mu) / std_dev


def downsample(input_signal, orig_freq, target_freq):
    """
    Downsamples the input signal, enforcing the Shannon-Nyquist criteria.

    Parameters
    ----------
    input_signal : TYPE: numpy.ndarray
        DESCRIPTION: The input signal to be downsampled.
    orig_freq : TYPE: int
        DESCRIPTION: The original sampling frequency of the signal.
    target_freq : TYPE: int
        DESCRIPTION: The target sampling frequency after downsampling.

    Raises
    ------
    ValueError
        DESCRIPTION: The final downsampling frequency cannot be higher than the
        original one.

    Returns
    -------
    downsampled_signal : TYPE: numpy.ndarray
        DESCRIPTION: The downsampled signal.

    """
    # Calculate the downsampling ratio
    ratio = target_freq / orig_freq

    # Ensure the ratio is less than 1 (downsampling)
    if ratio >= 1.0:
        raise ValueError("Target frequency must be less than the original"
                         + "frequency for downsampling.")

    # Calculate the greatest common divisor (GCD) to find the interpolation and
    # decimation factors
    gcd = np.gcd(int(target_freq), int(orig_freq))
    up = int(target_freq / gcd)
    down = int(orig_freq / gcd)

    # Design a low-pass filter to act as the antialiasing filter
    # Use firwin to create a FIR filter with a cutoff frequency at half the
    # target sampling rate
    numtaps = 101  # Number of taps in the FIR filter
    cutoff = target_freq / 2.0  # Cutoff frequency of the filter
    fir_filter = firwin(numtaps, cutoff=cutoff, fs=orig_freq)

    # Use resample_poly to apply the polyphase filtering and resampling
    downsampled_signal = resample_poly(
        input_signal, up, down, window=fir_filter)

    return downsampled_signal
