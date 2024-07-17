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

- butterworth_filter(data, filter_topology, order, fs, fc): filters the input
signal. The topology must be specified, as well as the order, sampling
frequency, and cutoff frequencies. The signal is forward-backward filtered,
removing the phase shifting, and doubling the order of the filters. It also
uses second order sections to ensure the stability of the filter.

- schmidt_spike_removal(original_signal, fs): This function removes the spikes
in a signal as done by Schmidt et al in the paper:
Schmidt, S. E., Holst-Hansen, C., Graff, C., Toft, E., & Struijk, J. J. (2010).
Segmentation of heart sound recordings by a duration-dependent
hidden Markov model.
Physiological Measurement, 31(4), 513-29.
This function was translated by ChatGPT.

- wavelet_denoise(data, decomposition_level, wavelet_family, risk_estimator,
                  shutdown_bands): This function filters and
denoises the inpud data based on wavelet decomposition. It requires a risk
estimation function to opperate. This function assumes that the input data is
standardized, thus, the noise standard deviation = 1

- val_SURE_threshold(X): computes the threshold for the wavelet denoise
function based on SURE. Function taken from
https://pyyawt.readthedocs.io/_modules/pyyawt/denoising.html#thselect

- universal_threshold(coefficients): computes the threshold for wavelet
denoise, this function allows for the implementation of VisuShrink denoising
approach
https://academic.oup.com/biomet/article-abstract/81/3/425/256924?redirectedFrom=fulltext


@author: Daniel Proaño-Guevara.

@creationDate: 2024-07-12

@version: 0.1
"""

# TODO: Homomorphic Envelogram.
# TODO: Hilbert Envelope.
# TODO: Wavelet Envelope.
# TODO: Power Spectral Density Envelope.

import numpy as np
from scipy.signal import firwin, resample_poly
from scipy import signal
import pywt


def resolution_normalization(data, resolution, centered_on_zero=True):
    """
    Normalize the data based on the known acquisition resolution.

    Parameters
    ----------
    data : numpy.ndarray
        Signal array to be normalized.
    resolution : int
        Bit depth in binary. Comes from the knowledge of the
        sampling resolution of the signal.
    centered_on_zero : Boolean, optional
        Enables centering the normalized value around zero. The
        default is True.

    Returns
    -------
    normalized_data : numpy.ndarray
        Data normalized between 0.5 and 0.5 based on the
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
    data : numpy.ndarray
        The input signal to be standardized.

    Returns
    -------
    numpy.ndarray
        The signal standardized.

    """
    mu = np.mean(data)
    std_dev = np.std(data)
    return (data - mu) / std_dev


def downsample(input_signal, orig_freq, target_freq):
    """
    Downsamples the input signal, enforcing the Shannon-Nyquist criteria.

    Parameters
    ----------
    input_signal : numpy.ndarray
        The input signal to be downsampled.
    orig_freq : int
        The original sampling frequency of the signal.
    target_freq : int
        The target sampling frequency after downsampling.

    Raises
    ------
    ValueError
        The final downsampling frequency cannot be higher than the
        original one.

    Returns
    -------
    downsampled_signal : numpy.ndarray
        The downsampled signal.

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


def butterworth_filter(data, filter_topology, order, fs, fc):
    """
    Filter the input data with a forward-backward structure.

    Parameters
    ----------
    data : numpy.ndarray
        Is the signal to be filtered.
    filter_topology : String
        The topology of the filter {'lowpass', 'highpass', 'bandpass',
                                    'bandstop'}.
    order : int
        The order of the filter.
    fs : int
        The sampling frequency in Hz.
    fc : TYPE: Int, [Int, Int]
        The cutoff frequency, or frequencies. For bandpass or bandstop, pass
    the coefficients between brackets [a, b].

    Returns
    -------
    numpy.ndarray
        The filtered signal.

    """
    coefficients = signal.butter(order, fc, btype=filter_topology,
                                 output="sos", fs=fs)
    return signal.sosfiltfilt(coefficients, data)


def schmidt_spike_removal(original_signal, fs):
    """
    Remove the spikes in a signal.

    Parameters:
    original_signal (numpy array): The original (1D) audio signal array
    fs (int): The sampling frequency (Hz)

    Returns:
    numpy array: The audio signal with any spikes removed.
    """
    # Find the window size (500 ms)
    windowsize = round(fs / 2)

    # Find any samples outside of an integer number of windows
    trailingsamples = len(original_signal) % windowsize

    # Reshape the signal into a number of windows
    sampleframes = np.reshape(original_signal[:-trailingsamples],
                              (windowsize, -1), order='F')

    # Find the MAAs
    MAAs = np.max(np.abs(sampleframes), axis=0)

    # While there are still samples greater than 3 * the median value of the
    # MAAs, then remove those spikes
    while np.any(MAAs > np.median(MAAs) * 3):
        # Find the window with the max MAA
        val = np.max(MAAs)
        window_num = np.argmax(MAAs)
        if np.size(window_num) > 1:
            window_num = window_num[0]

        # Find the position of the spike within that window
        val = np.max(np.abs(sampleframes[:, window_num]))
        spike_position = np.argmax(np.abs(sampleframes[:, window_num]))

        # Finding zero crossings (where there may not be actual 0 values, just a change from positive to negative)
        zero_crossings = np.abs(
            np.diff(np.sign(sampleframes[:, window_num]))) > 1
        zero_crossings = np.append(zero_crossings, 0)

        # Find the start of the spike, finding the last zero crossing before
        # spike position. If that is empty, take the start of the window
        # Find the last zero crossing before the spike position
        last_zero_crossing = np.max(
            np.where(zero_crossings[:spike_position])[0], initial=-1)

        # Convert to 1-based indexing by adding 1 (if necessary)
        spike_start = max(1, last_zero_crossing + 1)

        # Find the start of the spike, finding the last zero crossing before
        # spike position. If that is empty, take the start of the window

        # Assuming zero_crossings is a numpy array
        zero_crossings[:spike_position] = 0

        # Find the first non-zero element
        # np.argmax returns the index of the first occurrence of a condition
        first_nonzero = np.argmax(zero_crossings > 0)
        if zero_crossings[first_nonzero] == 0:
            # If no non-zero element is found, handle appropriately
            first_nonzero = len(zero_crossings)

        spike_end = min(first_nonzero, windowsize)

        # Set to zero
        sampleframes[spike_start:spike_end+1, window_num] = 0.0001
        # Recalculate MAAs
        MAAs = np.max(np.abs(sampleframes), axis=0)

    # Reshape the despiked signal back to 1D
    despiked_signal = np.reshape(sampleframes, -1, order='F')

    # Add the trailing samples back to the signal
    despiked_signal = np.concatenate((despiked_signal,
                                      original_signal[len(despiked_signal):]))

    return despiked_signal


def wavelet_denoise(data, decomposition_level, wavelet_family,
                    risk_estimator, shutdown_bands):
    """
    Wavelet denoising and filtering.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be processed.
    decomposition_level : int
        The level of wavelet decompositions.
    wavelet_family : string
        The wavelet family available in the pywt library.
    risk_estimator : function
        function that estimates the risk and sets the elimination threshold.
    shutdown_bands : array
        array of bands to be shut down.

    Returns
    -------
    numpy.ndarray
        The filtered and denoised data.

    """
    # Wavelet decomposition
    wv_coeff = pywt.wavedec(data, wavelet=wavelet_family,
                            level=decomposition_level)
    # Band denoising
    for band_index in range(len(wv_coeff)-1):
        # Compute lambda threshold for risk reduction
        lambda_threshold = risk_estimator(wv_coeff[band_index + 1])
        # Threshold the band
        wv_coeff[band_index + 1] = pywt.threshold(data, value=lambda_threshold,
                                                  mode='garrote')
    # Remove mean in approximation band
    wv_coeff[0] -= np.mean(wv_coeff[0])
    # Shutdown bands
    for band in shutdown_bands:
        wv_coeff[band][:] = 0
    # Reconstruct the signal
    return pywt.waverec(wv_coeff, wavelet=wavelet_family)


def val_SURE_threshold(X):
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


def universal_threshold(coefficients):
    """
    Calculate the universal threshold for risk reduction.

    Parameters
    ----------
    coefficients : numpy.ndarray
        The coefficients to estimate the risk from.

    Returns
    -------
    float
        The value of threshold.

    """
    n = len(coefficients)
    return np.sqrt(2 * np.log(n))
