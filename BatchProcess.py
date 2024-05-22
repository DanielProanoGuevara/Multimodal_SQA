# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:44:45 2024

@author: danie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import time
import pydub
import re
import pywt

from scipy import signal
from biosppy.signals import ecg


def merge_rows(group):
    # Initialize merged_row with the first row in the group
    merged_row = group.iloc[0].copy()
    # Iterate over columns
    for col in group.columns[2:]:
        # If there is a non-null value in the current column, update merged_row
        if not group[col].isnull().all():
            merged_row[col] = group[col].dropna().iloc[0]
    return merged_row


def minmaxNorm(data):
    return (data - np.min(data))/(np.max(data)-np.min(data))

def bandPassButterworth(data, a, b, rate, decimate=1):
    coefficients = signal.butter(64, [a, b], btype="bandpass", output="sos", fs=rate)
    filt = signal.sosfiltfilt(coefficients, data)
    return signal.decimate(filt, decimate)

def outlierRemoval(data):
    return np.clip(data, np.median(data)-3*np.std(data),
                    np.median(data)+3*np.std(data))


def waveletDenoise(signal,family, level):
    wv_coeffs = pywt.wavedec(signal, wavelet=family, level=level)
    dev = np.median(abs(wv_coeffs[-1]))/0.674
    for i in range(len(wv_coeffs)):
        wv_coeffs[i] = softThreshold (wv_coeffs[i], dev)
    return pywt.waverec(wv_coeffs, wavelet=family)

def softThreshold (signal, dev):
    #lbda = abs(np.median(signal)) + dev*abs(np.std(signal)) # personal criteria
    #lbda = dev*np.sqrt(2*np.log(int(signal.size))) # sqtwolog criteria
    lbda = dev*np.sqrt(2*np.log(int(signal.size)*np.log2(int(signal.size)))) # pseudo rigsure criteria
    return np.sign(signal) * np.maximum(np.abs(signal) - lbda, 0)

def shannonEnergyEnvelope(signal, windowSize):
    windowSize = windowSize
    i = 0
    shannonEnvelope = []

    while i < signal.size - windowSize + 1:
        # Store elements from i to i+windowSize
        window = signal[i : i + windowSize]
        
        # Calculate the 2nd Order Shannon Energy
        with np.errstate(invalid='raise'):
            try:
                windowSE = -(1/windowSize) * np.sum((window**2)*np.log(window**2))
            except:
                windowSE = 0
        
        # create list
        shannonEnvelope.append(windowSE)
        
        i+= 1
    return np.pad(shannonEnvelope,(int(windowSize/2),int(windowSize/2)), 'constant', constant_values=(0,0))


def movingAverage(signal, window):
    N = window
    return np.convolve(signal, np.ones(N)/N, mode='valid')

def segmentECG(ECG):
    ECG_rate = 500
    ECG_n = minmaxNorm(ECG)
    ECG_BP = bandPassButterworth(ECG_n, 8, 50, ECG_rate)
    ECG_clip = outlierRemoval(ECG_BP)
    
    ECG_n = minmaxNorm(ECG_clip)

    ## detect R-peaks
    rPeaks = ecg.hamilton_segmenter(ECG_n, ECG_rate)
    # create zero vector
    ECG_peaks = np.zeros_like(ECG_n)

    for idx in rPeaks[0]:
        ECG_peaks[idx] = 1

    #250 ms parabole
    x = np.linspace(-62, 62, 125)
    s = - (x**2) + 3844

    # convolve with pulses to create a "probability zone"
    ECG_peaks = minmaxNorm(np.convolve(ECG_peaks, s, mode='same'))


    # Decimate ECG and peaks
    ECG_n = ECG_n[::10]
    ECG_peaks = ECG_peaks[::10]
    return ECG_peaks


def segmentPCG(PCG):
    PCG_rate = 8000
    PCG_n = minmaxNorm(PCG)
    PCG_BP = bandPassButterworth(PCG_n, 50, 250, PCG_rate, decimate=16)
    PCG_rate /= 16
    PCG_clip = outlierRemoval(PCG_BP)
    PCG_n =outlierRemoval(waveletDenoise(PCG_clip,'coif4', 5))

    ## 2nd order Shannon energy envelope
    envelopeSE = shannonEnergyEnvelope(PCG_n, 20)

    ## Normalize envelope and create lobes
    normSE = envelopeSE - np.mean(envelopeSE)
    ## Lobes
    lobSE = np.where(normSE > 0, normSE, 0)


    ## Compute threshold for SE
    dev = np.median(abs(lobSE))*0.5
    ## Apply soft threshold
    softSE = minmaxNorm( movingAverage(softThreshold (lobSE, dev), 10))

    # get peaks location
    PCG_peaks_loc, _ = signal.find_peaks(softSE, distance=100)

    # create zero vector
    PCG_peaks = np.zeros_like(PCG_n)
    
    #250 ms parabole
    x = np.linspace(-62, 62, 125)
    s = - (x**2) + 3844

    for idx in PCG_peaks_loc:
        PCG_peaks[idx] = 1
        
    # Convolve 250 ms bells
    PCG_peaks = minmaxNorm(np.convolve(PCG_peaks, s, mode='same'))
        

    PCG_n = minmaxNorm(PCG_n[::10])
    PCG_peaks = PCG_peaks[::10]
    return PCG_peaks


# Define the sample directory
directory_dataset = '../DatasetCHVNGE'
paths = []

data = []





# Get all the files inside the directory

for filename in glob.iglob(f'{directory_dataset}/*'):
    #paths.append(filename)
    # Check if the file has either mp3 or raw extension
    if filename.endswith(".mp3") or filename.endswith(".raw"):
        # Split the filename into parts based on inderscore
        parts = filename.split("_")
        # Extract patient number and auscultation point
        head = parts[0]
        temp = re.findall(r'\d+', head)
        patient = list(map(int, temp))[0]
        tail = parts[1]
        tail_parts = tail.split(".")
        auscultation_point = tail_parts[0]
        # Read the content of the file
        if filename.endswith(".mp3"):
            a = pydub.AudioSegment.from_mp3(filename)
            t = a.duration_seconds
            PCG = np.array(a.get_array_of_samples())
            
            ECG_rate = None
            ECG = None
        elif filename.endswith(".raw"):
            ECG = np.loadtxt(filename, delimiter=",", dtype=int)
            
            PCG = None
            PCG_rate = None
        # Apend data to the list
        data.append([patient, auscultation_point, ECG, PCG])
        
# Create a DataFrame from collectedd data
db_df = pd.DataFrame(data, columns=["Patient", "Auscultation_Point", "ECG", "PCG"])
df_merged = db_df.groupby(["Patient", "Auscultation_Point"]).apply(merge_rows)
df_merged.reset_index(drop=True,inplace = True)

#%% Segment ECG and PCG
#(use df.apply)
df_merged['ECG_Lobes'] = df_merged['ECG'].apply(segmentECG)
df_merged['PCG_Lobes'] = df_merged['PCG'].apply(segmentPCG)
df_merged.to_pickle("./processedDataBase.pkl")