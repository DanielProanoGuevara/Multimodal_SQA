"""
ECG and PCG feature extraction library.

@file: feature_extraction_lib.py

@coding: utf_8

@description: This module intends to contain all the feature extraction
functions necessary for ECG and PCG signals.

@functions:
-


@author: Daniel Proaño-Guevara.

@creationDate: 2024-07-19

@version: 0.1
"""


# TODO: Ask about using DWT for the envelop, same additive process or just use
# lowest frequency one.
# TODO: Hilbert Envelope.
# TODO: Power Spectral Density Envelope.

import numpy as np
from scipy.signal import firwin, resample_poly
from scipy import signal
from preprocessing_lib import downsample, min_max_norm
import pywt


def homomorphic_envelope(data, fs_inicial, fs_final, epsilon=0.01,
                         median_window=51):
    """
    Compute the homomorphic envelope.

    Parameters
    ----------
    data : numpy.ndarray
        Input signal to obtain the envelope.
    fs_inicial : int
        original sampling frequency.
    fs_final : int
        final sampling frequency.
    epsilon : float
        replace zero value, corrects logarithm error.
    median_window : int (odd)
        kernel size for median filter.

    Returns
    -------
    numpy.ndarray
        homomorphic envelope of passed data.

    """
    # energy_signal = data**2
    energy_signal = abs(data)
    energy = np.where(energy_signal == 0, epsilon, energy_signal)
    g = np.log(energy)
    envelope_log = signal.medfilt(g, median_window)
    envelope = np.exp(envelope_log)

    return min_max_norm(downsample(envelope, fs_inicial, fs_final))
    # return (envelope)


def c_wavelet_envelope(data, fs_inicial, fs_final, wv_family='morl',
                       interest_frequencies=[50, 200]):
    """
    Compute the wavelet envelope based on CWT.

    Parameters
    ----------
    data : numpy.ndarray
        Input signal to obtain the envelope.
    fs_inicial : int
        original sampling frequency.
    fs_final : int
        final sampling frequency.
    wv_family : String, optional
        Continuous wavelet family. The default is 'morl'.
    interest_frequencies : list, optional
        Pair frequencies that represent the interest range to be analyzed,
        first the lower frequency, last the highest frequency.
        The default is [50, 200].

    Returns
    -------
    numpy.ndarray
        CWT-based envelope of passed data.

    """
    # Create an array of nomralized frequencies of interest for the continuous
    # wavelet transform
    frequencies = np.geomspace(interest_frequencies[0],
                               interest_frequencies[1], num=20)/fs_inicial
    # Create the scales from the frequencies
    scales = pywt.frequency2scale(wv_family, frequencies)
    # Extract the wavelet decomposition with CWT
    decomposition, _ = pywt.cwt(data, scales, wv_family)
    # Obtain the family of envelopes
    envelopes = np.abs(decomposition)
    # Combine the envelopes into a unique one
    envelope = np.mean(envelopes, axis=0)
    return min_max_norm(downsample(envelope, fs_inicial, fs_final))
    # return envelope


def d_wavelet_envelope(data, fs_inicial, fs_final, wv_family='rbio3.9',
                       level=3):
    """
    Compute the wavelet envelope based on DWT.

    Parameters
    ----------
    data : numpy.ndarray
        Input signal to obtain the envelope.
    fs_inicial : int
        original sampling frequency.
    fs_final : int
        final sampling frequency.
    wv_family : String, optional
        Continuous wavelet family. The default is 'rbio3.9'.
    level : int, optional
        Decomposition level for the lowest-frequency detail. The default is 3.

    Returns
    -------
    numpy.ndarray
        DWT-based envelope of passed data.

    """
    wv_detail = pywt.downcoef(part='d', data=data, wavelet=wv_family,
                              level=level, mode='zero')
    envelope = np.abs(wv_detail)
    return min_max_norm(downsample(envelope, fs_inicial/(2**level), fs_final))


def hilbert_envelope(data, fs_inicial, fs_final):
    """
    Compute the Hilbert-transform-based envelope.

    Parameters
    ----------
    data : numpy.ndarray
        Input signal to obtain the envelope.
    fs_inicial : int
        original sampling frequency.
    fs_final : int
        final sampling frequency.

    Returns
    -------
    numpy.ndarray
        Hilbert transform-based envelope of passed data.

    """
    analytic_signal = signal.hilbert(data)
    envelope = np.abs(analytic_signal)
    return min_max_norm(downsample(envelope, fs_inicial, fs_final))
