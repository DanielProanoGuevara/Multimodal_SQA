# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 14:26:40 2025

@author: Asus
"""

import numpy as np
from scipy.signal import hilbert, butter, filtfilt, find_peaks
from sqi_core_lib import sampen
from preprocessing_lib import power_spectral_density, bandpower_psd
from feature_extraction_lib import homomorphic_envelope, hilbert_envelope

def _butterworth_low_pass_filter(x, order, cutoff_hz, fs):
    nyq = 0.5 * fs
    wn = cutoff_hz / nyq
    b, a = butter(order, wn, btype='low')
    return filtfilt(b, a, x)

def _zerocrossings(siggrad):
    g = np.asarray(siggrad, dtype=float)
    g[g == 0] = 1e-12
    sg = np.sign(g)
    idx = np.where(np.diff(sg) != 0)[0] + 1
    dirs = []
    for i in idx:
        s0 = sg[i-1]; s1 = sg[i]
        if s0 < 0 and s1 > 0: dirs.append(1)   # minimum
        elif s0 > 0 and s1 < 0: dirs.append(-1) # maximum
        else: dirs.append(0)
    return np.array(idx, dtype=int), np.array(dirs, dtype=int)

def _pcg_truncated_acf_from_envelope(audio_data, fs):
    x = np.asarray(audio_data, dtype=float)
    N = x.size
    env = np.abs(hilbert(x))
    env_centered = env - np.mean(env)
    acf_full = np.correlate(env_centered, env_centered, mode='full')
    acf = acf_full[N-1:]
    acf = acf / acf[0]
    acf_lp = _butterworth_low_pass_filter(acf, order=1, cutoff_hz=15.0, fs=fs)
    siggrad = np.diff(acf_lp)
    crosspts, crossdirs = _zerocrossings(siggrad)
    if crosspts.size == 0:
        return acf_lp, acf_lp

    first_point = crosspts[0]
    c = 0
    min_samp = int(0.2 * fs)
    while first_point < min_samp and c+1 < crosspts.size:
        c += 1
        first_point = crosspts[c]

    target = crosspts[0] + int(5 * fs)
    place = np.argmin(np.abs(crosspts - target))
    second_point = crosspts[place]
    while place < crossdirs.size and crossdirs[place] == -1:
        place += 1
        if place < crosspts.size:
            second_point = crosspts[place]
        else:
            break

    first_point = max(first_point, 0)
    second_point = min(second_point, acf_lp.size-1)
    if second_point <= first_point:
        truncated = acf_lp
    else:
        truncated = acf_lp[first_point:second_point]
    return truncated, acf_lp

def se_sqi_pcg(audio_data, fs, M=2, r=None):
    """
    Springer-style seSQI = SampEn(M, r, N) of truncated ACF of PCG envelope.
    """
    tacf, _ = _pcg_truncated_acf_from_envelope(audio_data, fs)
    if tacf.size == 0:
        return np.nan
    if r is None:
        r_eff = 0.2 * np.std(tacf, ddof=0)
    else:
        r_eff = r
    return sampen(tacf, M, r_eff, sflag=0)
