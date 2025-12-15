# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 13:29:19 2025

@author: Asus
"""

# sqi_core_lib.py
import numpy as np
import tensorflow as tf
from scipy.signal import hilbert
from preprocessing_lib import power_spectral_density, bandpower_psd

# ---------- Sample Entropy (PhysioNet-style) ----------

def _embed_1d(y, m):
    y = np.asarray(y, dtype=float)
    N = y.size
    if N < m:
        return np.empty((0, m))
    return np.array([y[i:i+m] for i in range(N - m + 1)])

def _phi_chebyshev(X, r):
    X = np.asarray(X, dtype=float)
    K = X.shape[0]
    if K <= 1:
        return 0.0
    count = 0
    for i in range(K - 1):
        xi = X[i]
        diffs = np.max(np.abs(X[i+1:] - xi), axis=1)
        count += np.sum(diffs <= r)
    total_pairs = K * (K - 1) / 2.0
    if total_pairs == 0 or count == 0:
        return 0.0
    return (2.0 * count) / (K * (K - 1))

def sampen_vector(y, M, r, sflag=1):
    """
    PhysioNet-like SampEn implementation.
    Returns e[0..M-1], where e[m] ~ SampEn(m, r, N).
    """
    y = np.asarray(y, dtype=float).ravel()
    if sflag > 0:
        y = y - np.mean(y)
        s = np.sqrt(np.mean(y**2))
        if s > 0:
            y /= s

    e = np.empty(M, dtype=float)
    if M > 0:
        e[0] = np.nan
    for m in range(1, M):
        Xm  = _embed_1d(y, m)
        Xm1 = _embed_1d(y, m+1)
        Um  = _phi_chebyshev(Xm,  r)
        Um1 = _phi_chebyshev(Xm1, r)
        if Um <= 0.0 or Um1 <= 0.0:
            e[m] = np.inf
        else:
            e[m] = -np.log(Um1 / Um)
    return e

def sampen(y, M, r, sflag=1):
    """
    Convenience: return SampEn(M, r, N).
    """
    e = sampen_vector(y, M+1, r, sflag=sflag)
    return e[M]

def flatline_fraction(signal, fs, window_sec=0.2, tol=1e-4):
    """
    Fraction of time the signal is effectively flat (|Δx| < tol over a window).
    Used for ECG fSQI and can be reused elsewhere.
    """
    x = np.asarray(signal, dtype=float)
    dx = np.abs(np.diff(x))
    is_flat = dx < tol

    win = int(window_sec * fs)
    if win <= 1:
        return np.mean(is_flat)

    # Convolve with ones to find long flat stretches
    kernel = np.ones(win, dtype=float)
    flat_len = np.convolve(is_flat.astype(float), kernel, mode='same')
    flat_long = flat_len >= win
    return flat_long.mean()

def mfcc_band_energy_stats_tf(signal, fs, band_targets,
                               n_mels=14, fmin=50.0, fmax=1000.0):
    """
    Compute log-mel energy stats for given mel bands using TensorFlow.

    Parameters
    ----------
    signal : np.ndarray
        The input 1D PCG signal.
    fs : int
        Sampling rate.
    band_targets : list of dict
        Each dict must have: 'band', 'stat' (callable), and 'name'.
    n_mels : int
        Number of mel bands.
    fmin : float
        Minimum frequency of mel filters.
    fmax : float
        Maximum frequency of mel filters.

    Returns
    -------
    dict
        Dictionary of feature name -> scalar value.
    """
    audio_tensor = tf.convert_to_tensor(signal, dtype=tf.float32)
    audio_tensor = tf.expand_dims(audio_tensor, 0)  # batch dim

    frame_length = int(0.025 * fs)
    frame_step = int(0.010 * fs)
    fft_length = 2 ** int(np.ceil(np.log2(frame_length)))

    stft = tf.signal.stft(
        audio_tensor,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=tf.signal.hamming_window,
        pad_end=False
    )
    power_spec = tf.abs(stft) ** 2

    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=fft_length // 2 + 1,
        sample_rate=fs,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax
    )
    mel_spec = tf.matmul(power_spec, mel_weights)
    log_mel = tf.math.log(mel_spec + 1e-6)  # Avoid log(0)
    log_mel = log_mel.numpy()[0].T  # shape: (n_mels, n_frames)

    result = {}
    for item in band_targets:
        values = log_mel[item['band'], :]
        result[item['name']] = item['stat'](values)
    return result
