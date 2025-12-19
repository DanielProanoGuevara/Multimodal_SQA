# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 14:41:57 2025

@author: danie
"""
import numpy as np
from scipy.stats import skew, kurtosis
from sqi_core_lib import flatline_fraction
from preprocessing_lib import (power_spectral_density,
                               bandpower_psd,)
import neurokit2 as nk


def bSQI(ecg_signal, fs, tolerance_ms=50):
    """
    Beat Signal Quality Index (bSQI) from Clifford et al. 2012.
    Compares R-peaks detected by wQRS and eplimited detectors.

    Parameters
    ----------
    ecg_signal : array-like
        Raw ECG signal.
    fs : int
        Sampling frequency in Hz.
    tolerance_ms : float
        Matching window (ms) within which R-peaks are considered matched.

    Returns
    -------
    float
        bSQI score: proportion of wQRS beats that match eplimited beats.
    """
    tolerance_samples = int((tolerance_ms / 1000.0) * fs)

    # Use NeuroKit's preprocessing
    ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=fs)

    # Detect peaks using both detectors
    wqrs_peaks = nk.ecg_findpeaks(ecg_clean, sampling_rate=fs, method="zong2003")[
        "ECG_R_Peaks"]
    hamilton_peaks = nk.ecg_findpeaks(
        ecg_clean, sampling_rate=fs, method="hamilton2002")["ECG_R_Peaks"]

    if len(wqrs_peaks) == 0:
        return 0.0

    # Count matches within tolerance
    matched = 0
    hamilton_set = set(hamilton_peaks)
    for w_peak in wqrs_peaks:
        if any(abs(w_peak - h_peak) <= tolerance_samples for h_peak in hamilton_set):
            matched += 1

    return matched / len(wqrs_peaks)


def pSQI(ecg_signal, fs):
    """
    Relative power in the QRS band (5–15 Hz) to broadband (5–40 Hz).
    """
    f, Pxx = power_spectral_density(ecg_signal, fs)
    qrs_power = bandpower_psd(f, Pxx, (5, 15))
    total_power = bandpower_psd(f, Pxx, (5, 40))
    return 0.0 if total_power == 0 else qrs_power / total_power


def sSQI(ecg_signal):
    """
    Skewness of the ECG signal.
    """
    return float(skew(ecg_signal))


def kSQI(ecg_signal):
    """
    Kurtosis of the ECG signal.
    """
    return float(kurtosis(ecg_signal))


def fSQI(ecg_signal, fs, window_sec=0.2, tol=1e-4):
    """
    Percentage of the signal that appears to be flat.
    Uses the moving difference-based approach.
    """
    return flatline_fraction(ecg_signal, fs, window_sec, tol)


def basSQI(ecg_signal, fs):
    """
    Relative power in the baseline (0–1 Hz) to total (0–40 Hz), subtracted from 1.
    """
    f, Pxx = power_spectral_density(ecg_signal, fs)
    baseline_power = bandpower_psd(f, Pxx, (0, 1))
    total_power = bandpower_psd(f, Pxx, (0, 40))
    return 0.0 if total_power == 0 else 1 - (baseline_power / total_power)
