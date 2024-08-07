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

# TODO: Homomorphic Envelogram.
# TODO: Hilbert Envelope.
# TODO: Wavelet Envelope.
# TODO: Power Spectral Density Envelope.

import numpy as np
from scipy.signal import firwin, resample_poly
from scipy import signal
from preprocessing_lib import downsample


def homomorphic_envelope(data, fs_inicial, fs_final):
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

    Returns
    -------
    numpy.ndarray
        homomorphic envelope of passed data.

    """

    # energy_signal = data**2
    energy_signal = abs(data)
    energy = np.where(energy_signal == 0, 0.01, energy_signal)
    g = np.log(energy)
    envelope_log = signal.medfilt(g, 51)
    envelope = np.exp(envelope_log)

    # return downsample(envelope, fs_inicial, fs_final)
    return envelope
