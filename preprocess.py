# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:42:15 2024

@author: danie
"""
#%% Import libraries

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pywt
import pydub
import time
import sounddevice as sd
from scipy import signal
from biosppy.signals import ecg

#%% Functions, objects, classes

## Visualization Functions
def visualizeSpectro(data, rate, title):
    y_max = np.max(data)
    
    fft = np.fft.fft(data)
    freq = np.fft.fftfreq(data.shape[-1])
    x = freq[0:int(data.shape[-1]/2 + 1)]*rate
    y = fft.real[0:int(data.shape[-1]/2 + 1)]
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(data)
    plt.title(title)
    plt.grid()
    plt.xlabel("Samples")
    plt.subplot(2,1,2)
    plt.plot(x, y)
    plt.ylim([0,y_max])
    plt.xlim([0,rate/2])
    plt.grid()
    plt.xlabel("Frequency [Hz]")
    plt.tight_layout()
    plt.show()
    
def softThreshold (signal, dev):
    lbda = dev * np.sqrt(2 * np.log(len(signal))) # Universal threshold calculator
    return np.sign(signal) * np.maximum(np.abs(signal) - lbda, 0)

def waveletDenoise(signal,family, level):
    wv_coeffs = pywt.wavedec(signal, wavelet=family, level=level)
    dev = np.median(abs(wv_coeffs[-2]))/0.674 # Estimate noise level MAD
    for i in range(len(wv_coeffs)):
        wv_coeffs[i] = softThreshold (wv_coeffs[i], dev)
    return pywt.waverec(wv_coeffs, wavelet=family)

def waveletDenoise2(signal,family, level):
    wv_coeffs = pywt.wavedec(signal, wavelet=family, level=level)
    lbda = np.sqrt(2*np.log(len(signal)))
    wv_coeffs[1:] = [pywt.threshold(c, lbda, mode='soft') for c in wv_coeffs[1:]]
    return pywt.waverec(wv_coeffs, wavelet=family)

def playSound(signal, fr, t):
    sd.play(signal,fr)
    time.sleep(t)
    sd.stop()


#%% Import signals

# Declare files path
PCG_path = r"../DatasetCHVNGE/16_TV.mp3"
ECG_path = r"../DatasetCHVNGE/16_TV.raw"

## Import PCG
a = pydub.AudioSegment.from_mp3(PCG_path)
PCG_rate = a.frame_rate
t = a.duration_seconds
PCG = np.array(a.get_array_of_samples())
PCG_bit_width = 16

## Import ECG
ECG = np.loadtxt(ECG_path, delimiter=",", dtype=int)
ECG_rate = 500
ECG_bit_width = 12
#%% ECG Processing

## Normalize full-scale
ECG_v = (ECG - 2047) /4095 #int 12 bits (scale -0.5;0.5)
visualizeSpectro(ECG_v, ECG_rate, 'ECG normalized full-scale')

## Wavelet after FIR


## Linear phase FIR
# Calculate FIR coefficients with window method
# Define the filter specifications
numtaps = 251                  # Number of taps in the filter
cutoff = [0.004, 0.6]          # Passband frequencies (normalized, from 0 to 1, where 1 is the Nyquist frequency)
window = 'hamming'             # Window type

taps = signal.firwin(numtaps, cutoff, pass_zero=False, window=window)

#ECG_filt = np.convolve(ECG_v, taps, mode='full')[250:-250] # Trim convolution overhead
ECG_filt = np.convolve(ECG_v, taps, mode='same')
visualizeSpectro(ECG_filt, ECG_rate, 'ECG FIR filtered')



ECG_filt_db4 = waveletDenoise(ECG_filt,'db4', 5)
visualizeSpectro(ECG_filt_db4, ECG_rate, 'ECG db4 after FIR')

# err1 = ECG_v - ECG_filt_db4[:np.size(ECG_v)]
# plt.figure()
# plt.plot(err1)
# plt.grid()


# err = ECG_v - ECG_filt
# plt.figure()
# plt.plot(err)
# plt.grid()

## FIR after wavelet
ECG_db4 = waveletDenoise(ECG_v,'db4', 5)
visualizeSpectro(ECG_db4, ECG_rate, 'ECG only db4')

#ECG_db4_filt = np.convolve(ECG_db4, taps, mode='full')[250:-250] # Trim convolution overhead
ECG_db4_filt = np.convolve(ECG_db4, taps, mode='same')
visualizeSpectro(ECG_db4_filt, ECG_rate, 'ECG FIR after db4')

# err2 = ECG_v - ECG_db4_filt[:np.size(ECG_v)]
# plt.figure()
# plt.plot(err2)
# plt.grid()

