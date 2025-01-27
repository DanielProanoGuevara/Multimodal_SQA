# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:42:15 2024

@author: danie
"""
# %% Import libraries

import matplotlib.pyplot as plt
import numpy as np
import pywt
import pydub
import time
import sounddevice as sd
from scipy import signal
from scipy.io.wavfile import write

# %% Functions, objects, classes

# Visualization Functions


def visualizeSpectro(data, rate, title):
    y_max = np.max(data)

    fft = np.fft.fft(data)
    freq = np.fft.fftfreq(data.shape[-1])
    x = freq[0:int(data.shape[-1]/2 + 1)]*rate
    y = fft.real[0:int(data.shape[-1]/2 + 1)]
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(data)
    plt.title(title)
    plt.grid()
    plt.xlabel("Samples")
    plt.subplot(2, 1, 2)
    plt.plot(x, y)
    plt.ylim([0, y_max])
    plt.xlim([0, rate/2])
    plt.grid()
    plt.xlabel("Frequency [Hz]")
    plt.tight_layout()
    plt.show()


def visualizeStacked(x1, x2):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x1)
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(x2)
    plt.grid()
    plt.tight_layout()
    plt.show


def minmaxNorm2(data):
    return (2*(data - np.min(data))/(np.max(data)-np.min(data)))-1


def softThreshold(signal, dev):
    # Universal threshold calculator
    lbda = dev * np.sqrt(2 * np.log(len(signal)))
    return np.sign(signal) * np.maximum(np.abs(signal) - lbda, 0)


def universal(coeffs, sigma):
    n = len(coeffs)
    return sigma * np.sqrt(2*np.log(n))


def normalshrink_lambda(coeffs, sigma):
    # Number of wavelet coefficients
    n = len(coeffs)
    beta = np.sqrt(np.log10(n/9))
    sigma_y = np.var(coeffs)

    lambda_normal = beta * (sigma ** 2) / sigma_y

    return lambda_normal


def residuals(filtered, raw):
    return raw - filtered[:len(raw)]


def ecgWVDenoise(signal, family, level, threshold):
    wv_coeffs = pywt.wavedec(signal, wavelet=family, level=level)
    sigma = np.median(abs(wv_coeffs[-2]))/0.674
    for i in range(len(wv_coeffs)-1):
        lbda = threshold(wv_coeffs[i+1], sigma)
        wv_coeffs[i+1] = softThreshold(wv_coeffs[i+1], lbda)
    wv_coeffs[0][:] = 0
    return pywt.waverec(wv_coeffs, wavelet=family)


def pcgWVDenoise(signal, threshold):
    wv_coeffs = pywt.wavedec(signal, wavelet='coif4', level=5)
    sigma = np.median(abs(wv_coeffs[1]))/0.674
    for i in range(len(wv_coeffs)):
        lbda = threshold(wv_coeffs[i], sigma)
        wv_coeffs[i] = softThreshold(wv_coeffs[i], lbda)
    wv_coeffs[0] -= np.mean(wv_coeffs[0])
    wv_coeffs[-1][:] = 0
    wv_coeffs[-2][:] = 0
    wv_coeffs[-3][:] = 0
    return pywt.waverec(wv_coeffs, wavelet='coif4')


def playSound(signal, fr, t):
    sd.play(signal, fr)
    time.sleep(t)
    sd.stop()


def totalPNID(noise, raw):
    noise_power = signalPower(noise)
    raw_power = signalPower(raw)
    return 10 * np.log10(noise_power / raw_power)


def meanSquareError(filtered, raw):
    filtered = np.array(filtered)
    raw = np.array(raw)
    return np.mean((raw - filtered[:len(raw)]) ** 2)


def signalPower(signal):
    return np.mean(signal ** 2)


def bandPassButterworth(data, a, b, rate, decimate=1):
    coefficients = signal.butter(
        8, [a, b], btype="bandpass", output="sos", fs=rate)
    filt = signal.sosfiltfilt(coefficients, data)
    return filt[::decimate]


def shannonEnergy(signal):
    epsilon = 0.0001  # Small constant to prevent log(0)
    energy = signal ** 2
    se = -energy * np.log(energy + epsilon)
    return se


def smooth_signal(signal, window_len=50):
    hamming_window = np.hamming(window_len)
    return np.convolve(signal, hamming_window, mode='same')


def playback(data, rate, t):
    sd.play(data, rate)
    time.sleep(t)
    sd.stop()


def saveAudio(data, rate, name):
    sound = minmaxNorm2(data)
    write(name, rate, sound.astype(np.float32))

# %% Import signals


# Declare files path
PCG_path = r"../DatasetCHVNGE/98_AV.mp3"
ECG_path = r"../DatasetCHVNGE/98_AV.raw"

# Import PCG
a = pydub.AudioSegment.from_mp3(PCG_path)
PCG_rate = a.frame_rate
t = a.duration_seconds
PCG = np.array(a.get_array_of_samples())
PCG_bit_width = 16
PCG_resolution = (2 ** PCG_bit_width)-1

# Import ECG
ECG = np.loadtxt(ECG_path, delimiter=",", dtype=int)
ECG_rate = 500
ECG_bit_width = 12
ECG_resolution = (2 ** ECG_bit_width)-1

# plt.figure()
# plt.plot(PCG[55000:75000])

# visualizeStacked(ECG, PCG)
# %% ECG Processing

# Normalize full-scale
# int 12 bits (scale -0.5;0.5)
# ECG_v = (ECG - ECG_resolution/2) / (ECG_resolution)

# # Wavelet denoising
# ECG_wv_denoised = ecgWVDenoise(ECG_v, 'db4', 8, universal)
# e_ECG = residuals(ECG_wv_denoised, ECG_v)

# # Calculate ECG Noise Metrics
# mse_ECG = meanSquareError(ECG_wv_denoised, ECG_v)
# pnid_ECG = totalPNID(e_ECG, ECG_v)

# print("MSE ECG: ", mse_ECG, "\nNoise Distortion: ", pnid_ECG)

# %% Plots ECG

# #Plot Denoising
# fig, axs = plt.subplots(3,1, gridspec_kw={'height_ratios': [2, 2, 1]})
# fig.suptitle('ECG Filtering Process')
# axs[0].plot(ECG_v)
# axs[0].set_title('Raw ECG')
# axs[0].grid()
# axs[1].plot(ECG_wv_denoised)
# axs[1].set_title('Denoised ECG')
# axs[1].grid()
# axs[2].plot(e_ECG)
# axs[2].set_title('Estimated Noise')
# axs[2].grid()
# fig.supxlabel('Samples [500sps]')
# fig.supylabel('Amplitude [normalized]')

# #Plot Segmentation
# plt.figure()
# plt.plot(ECG_wv_denoised_d)
# plt.plot(ECG_peaks_d)
# plt.grid()
# plt.title('Segmented ECG')
# plt.xlabel('Samples [50sps]')
# plt.tight_layout()
# plt.show()

# %% PCG Processing

# Normalize full-scale
# PCG_v = (PCG) / (PCG_resolution)  # uint 16 bits (scale -0.5;0.5)
# # Signal Denoise
# PCG_wv_denoised = pcgWVDenoise(PCG_v, normalshrink_lambda)
# e_PCG = residuals(PCG_wv_denoised, PCG_v)

# # Calculate PCG Noise Metrics
# mse_PCG = meanSquareError(PCG_wv_denoised, PCG_v)
# pnid_PCG = totalPNID(e_PCG, PCG_v)

# print("MSE PCG: ", mse_PCG, "\nNoise Distortion: ", pnid_PCG)


# %%
visualizeStacked(ECG, PCG)
# # visualizeStacked(ECG_wv_denoised[1000:6000], PCG_wv_denoised[16000:96000])
# visualizeStacked(PCG_v, PCG_wv_denoised)
# visualizeStacked(ECG_v, ECG_wv_denoised)

# # playback(PCG_wv_denoised, PCG_rate, t)

# saveAudio(PCG_wv_denoised, PCG_rate, "48_TV_clean.wav")
