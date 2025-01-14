# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:16:31 2024

@author: danie
"""

# %%
# Imports

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import pywt
import pydub
import time
import wfdb
import sounddevice as sd
from scipy import signal
import preprocessing_lib as pplib
import feature_extraction_lib as ftelib
import file_process_lib as importlib

# import os

# import librosa
# import logging
# import numpy as np
# import pandas as pd
import scipy.io
# import scipy.signal
# import re

# import pickle

from scipy.io import wavfile
# import tensorflow as tf
# # from tqdm import tqdm
# import matplotlib.pyplot as plt


import copy

# %%

# pcg_dir = r"..\DatasetCHVNGE\5_AV.mp3"

# # Import audio file
# a = pydub.AudioSegment.from_mp3(pcg_dir)
# samplerate = a.frame_rate
# PCG = np.array(a.get_array_of_samples())
# PCG_bit_width = 16
# PCG_resolution = (2 ** PCG_bit_width)-1
# # Normalize full-scale
# original_data = (PCG) / (PCG_resolution)  # uint 16 bits (scale -0.5;0.5)

# data = np.copy(original_data)
# z_norm = pplib.z_score_standardization(data)

# # Resample 1kHz
# resample = pplib.downsample(z_norm, samplerate, 1000)

# # Schmidt despiking
# despiked_signal = pplib.schmidt_spike_removal(resample, 1000)

# despiked2 = np.copy(despiked_signal)

# # wavelet denoising
# wavelet_denoised = pplib.wavelet_denoise(
#     despiked_signal, 5, wavelet_family='coif4', risk_estimator=pplib.val_SURE_threshold, shutdown_bands=[-1, 1, 2])


# # Butterworth bandpass filtering
# filtered_wavelet = pplib.butterworth_filter(
#     wavelet_denoised, 'bandpass', 4, 1000, [15, 450])
# filtered_just_butt = pplib.butterworth_filter(
#     despiked2, 'bandpass', 4, 1000, [15, 450])
# # %% Plots Pre-processing

# # Subplots
# fig1, axs = plt.subplots(6, figsize=(7, 8))
# fig1.suptitle('ULSGE 5 AV (Quality = 3) Preprocessing Steps')
# axs[0].plot(data)
# axs[0].set_title('Raw Signal')
# axs[0].grid()

# axs[1].plot(z_norm)
# axs[1].set_title('Standardized Signal')
# axs[1].grid()

# axs[2].plot(despiked_signal)
# axs[2].set_title('Schmidt-Despiked Signal')
# axs[2].grid()

# axs[3].plot(wavelet_denoised)
# axs[3].set_title('Only Wavelet Denoised Signal')
# axs[3].grid()

# axs[4].plot(filtered_just_butt)
# axs[4].set_title('Only Butterworth Filtered Signal')
# axs[4].grid()

# axs[5].plot(filtered_wavelet)
# axs[5].set_title('Wavelet Denoised and Butterworth Filtered Signal')
# axs[5].grid()

# for ax in axs:
#     ax.set_xticks([])
#     # ax.set_yticks([])
# fig1.tight_layout()


# # Overlapping
# fig2, ax_ov = plt.subplots(1, figsize=(7, 5.25))
# fig2.suptitle('ULSGE 5 AV (Quality = 3) Preprocessing Strategies Comparison')
# ax_ov.plot(filtered_just_butt[20000:26000],
#            label='Only Butterworth Filtered Signal')
# ax_ov.plot(wavelet_denoised[20000:26000], label='Only Wavelet Denoised Signal')
# ax_ov.plot(filtered_wavelet[20000:26000],
#            label='Wavelet Denoised and Butterworth Filtered Signal')
# ax_ov.grid()
# ax_ov.set_xticks([])
# ax_ov.legend()
# fig2.tight_layout()

# # %% Envelopes
# filtered_pcg = np.copy(filtered_wavelet)
# # Reference PCG
# resample_ref_pcg = pplib.downsample(filtered_pcg, 1000, 50)

# # Homomorphic Envelope
# homomorphic = ftelib.homomorphic_envelope(
#     filtered_pcg, 1000, 50)

# # CWT Scalogram Envelope
# cwt_morl = ftelib.c_wavelet_envelope(filtered_pcg, 1000, 50,
#                                      interest_frequencies=[40, 200])

# cwt_mexh = ftelib.c_wavelet_envelope(
#     filtered_pcg, 1000, 50, wv_family='mexh',
#     interest_frequencies=[40, 200])

# # Hilbert Envelope
# hilbert_env = ftelib.hilbert_envelope(filtered_pcg, 1000, 50)

# # %% Plots Envelopes

# # Subplots
# # Subplots
# fig, axs = plt.subplots(4, figsize=(7, 8))
# fig.suptitle('ULSGE 5 AV (Quality = 3) Envelopes')
# axs[0].plot(homomorphic)
# axs[0].set_title('Homomorphic Envelope')
# axs[0].grid()

# axs[1].plot(cwt_morl)
# axs[1].set_title('Morlet Wavelet Scalogram Envelope')
# axs[1].grid()

# axs[2].plot(cwt_mexh)
# axs[2].set_title('Mexican Hat Wavelet Scalogram Envelope')
# axs[2].grid()

# axs[3].plot(hilbert_env)
# axs[3].set_title('Hilbert Envelope')
# axs[3].grid()

# for ax in axs:
#     ax.set_xticks([])
#     # ax.set_yticks([])
# fig.tight_layout()


# # Overlapping
# fig, ax = plt.subplots(1, figsize=(7, 5.25))
# fig.suptitle('ULSGE 5 AV (Quality = 3) Envelopes Comparison')
# ax.plot(cwt_morl[1000:1300], label='Morlet Wavelet Scalogram Envelope')
# ax.plot(cwt_mexh[1000:1300], label='Mexican Hat Wavelet Scalogram Envelope')
# ax.plot(hilbert_env[1000:1300], label='Hilbert Envelope')
# ax.plot(homomorphic[1000:1300], label='Homomorphic Envelope')
# ax.grid()
# ax.set_xticks([])
# ax.legend()
# fig.tight_layout()

# %% Denoising
# Import and analyze the dataset

# directory = r'../LUDB/data/2'

# # Read as record
# record = wfdb.rdrecord(directory)
# wfdb.plot_wfdb(record=record, title="Record 1 from LUDB")

# # Read only signals
# signals, fields = wfdb.rdsamp(directory, channels=[1, 3, 5, 6])

# # Read annotations
# ann = wfdb.rdann(directory, extension="i")
# wfdb.plot_wfdb(annotation=ann)

# # Plot annotations on top of signal
# wfdb.plot_wfdb(record=record, annotation=ann,
#                title="I lead annotated (not differentiated)")

# # indices where the annotation is applied
# annotation_index = ann.sample

# # symbol order of the annotations
# annotation_vector = ann.symbol
