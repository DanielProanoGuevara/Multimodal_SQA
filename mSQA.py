#%% Import libraries

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pywt
import pydub
import time
from scipy import signal
from biosppy.signals import ecg

#%% Functions, objects, classes

## Visualization Functions
def visualizeSpectro(data, rate):
    y_max = np.max(data)
    
    fft = np.fft.fft(data)
    freq = np.fft.fftfreq(data.shape[-1])
    x = freq[0:int(data.shape[-1]/2 + 1)]*rate
    y = fft.real[0:int(data.shape[-1]/2 + 1)]
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(data)
    plt.grid()
    plt.xlabel("Samples")
    plt.subplot(2,1,2)
    plt.plot(x, y)
    plt.ylim([0,y_max])
    plt.grid()
    plt.xlabel("Frequency [Hz]")
    plt.tight_layout()
    plt.show()
 
    
def visualizeStacked(x1, x2):
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(x1)
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(x2)
    plt.grid()
    plt.tight_layout()
    plt.show
## Utilitary functions
def minmaxNorm(data):
    return (data - np.min(data))/(np.max(data)-np.min(data))

def minmaxNorm2(data):
    return (2*(data - np.min(data))/(np.max(data)-np.min(data)))-1

def bandPassButterworth(data, a, b, rate, decimate=1):
    coefficients = signal.butter(64, [a, b], btype="bandpass", output="sos", fs=rate)
    filt = signal.sosfiltfilt(coefficients, data)
    return signal.decimate(filt, decimate)

def outlierRemoval(data):
    return np.clip(data, np.median(data)-3*np.std(data),
                    np.median(data)+3*np.std(data))

def softThreshold (signal, dev):
    #lbda = abs(np.median(signal)) + dev*abs(np.std(signal)) # personal criteria
    #lbda = dev*np.sqrt(2*np.log(int(signal.size))) # sqtwolog criteria
    lbda = dev*np.sqrt(2*np.log(int(signal.size)*np.log2(int(signal.size)))) # pseudo rigsure criteria
    return np.sign(signal) * np.maximum(np.abs(signal) - lbda, 0)

def waveletDenoise(signal,family, level):
    wv_coeffs = pywt.wavedec(signal, wavelet=family, level=level)
    dev = np.median(abs(wv_coeffs[-1]))/0.674
    for i in range(len(wv_coeffs)):
        wv_coeffs[i] = softThreshold (wv_coeffs[i], dev)
    return pywt.waverec(wv_coeffs, wavelet=family)

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

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]


def cross_corr(x,y):
    result = np.correlate(x, y, mode='full')
    return result[result.size//2:]

def cross_corr_back(x,y):
    result = np.correlate(x, y, mode='full')
    return result[:result.size//2]

def RMS(x):
    return np.sqrt(np.mean(x**2))

def harmonic_mean(a, b):
    return (2*a*b)/(a+b)

#%% Import signals

# Declare files path
PCG_path = r"../DatasetCHVNGE/7_TV.mp3"
ECG_path = r"../DatasetCHVNGE/7_TV.raw"

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

### Visualize
# visualizeSpectro(ECG, ECG_rate)
# visualizeSpectro(PCG, PCG_rate)

visualizeStacked(ECG[:3000], PCG[:48000])

#%% Preprocess common
## minmax normalization
ECG_n = minmaxNorm(ECG)
PCG_n = minmaxNorm(PCG)

# visualizeStacked(ECG_n, PCG_n)

## bandpass filtering and resampling
ECG_BP = bandPassButterworth(ECG_n, 8, 50, ECG_rate)
PCG_BP = bandPassButterworth(PCG_n, 50, 250, PCG_rate, decimate=16)
PCG_rate /= 16

# visualizeSpectro(ECG_BP, ECG_rate)
# visualizeSpectro(PCG_BP, PCG_rate)

## Clipping
ECG_clip = outlierRemoval(ECG_BP)
PCG_clip = outlierRemoval(PCG_BP)

# visualizeStacked(ECG_clip, PCG_clip)

#%% ECG feature extraction
## ECG renormalization
ECG_n = minmaxNorm(ECG_clip)

## detect R-peaks
rPeaks = ecg.hamilton_segmenter(ECG_n, ECG_rate)
# create zero vector
ECG_peaks = np.zeros_like(ECG_n)

for idx in rPeaks[0]:
    ECG_peaks[idx] = 1


# normal bell
# mu = 0
# variance = 1
# sigma = math.sqrt(variance)
# x = np.linspace(mu - 4*sigma, mu + 4*sigma, 10)
# s = stats.norm.pdf(x, mu, sigma)

#250 ms parabole
x = np.linspace(-62, 62, 125)
s = - (x**2) + 3844

# convolve with pulses to create a "probability zone"
ECG_peaks = minmaxNorm(np.convolve(ECG_peaks, s, mode='same'))


# Decimate ECG and peaks
# ECG_n = signal.decimate(ECG_n, 10)
# ECG_peaks = signal.decimate(ECG_peaks, 10)
ECG_n = ECG_n[::10]
ECG_peaks = ECG_peaks[::10]


# visualizeStacked(ECG_n, ECG_peaks)
#%% PCG preprocessing and feature extraction

## Wavelet denoising
PCG_n =outlierRemoval(waveletDenoise(PCG_clip,'coif4', 5))
# visualizeStacked(PCG_clip, PCG_n)

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

for idx in PCG_peaks_loc:
    PCG_peaks[idx] = 1
    
# Convolve 250 ms bells
PCG_peaks = minmaxNorm(np.convolve(PCG_peaks, s, mode='same'))
    
# PCG_n = minmaxNorm(signal.decimate(PCG_n, 10))
# PCG_peaks = signal.decimate(PCG_peaks, 10)

PCG_n = minmaxNorm(PCG_n[::10])
PCG_peaks = PCG_peaks[::10]

  

# visualizeStacked(PCG_n, PCG_peaks)

#%% Combine/Compare R-peaks with S1-S2 peaks
# visualizeStacked(ECG_n, PCG_n)
# visualizeStacked(ECG_peaks, PCG_peaks)

correlation_coeff = np.corrcoef(ECG_peaks, PCG_peaks)[0][1]

#%% Dummy comparison
overlap = np.where(ECG_peaks > 0, PCG_peaks, 0)

# plt.figure()
# plt.plot(overlap)
# plt.grid()
# plt.tight_layout()

rms = RMS(overlap)
autocorr_coeff = np.corrcoef(overlap)

area_score = np.trapz(overlap)
print('area= ',area_score, ' corrCoef = ', correlation_coeff, ' RMS = ', rms)


#%% Resiudes analysis based on dummy


residues = np.where(ECG_peaks > 0, 0, PCG_peaks)

# plt.figure()
# plt.plot(residues)
# plt.grid()
# plt.tight_layout()

residues_area = np.trapz(residues)

rms_residues = RMS(residues)

relation_area = area_score/residues_area


relation_rms = rms/rms_residues

print('\n residues area = ', residues_area, ' residues RMS = ', rms_residues, 
      '\n relation area = ', relation_area, ' relation RMS = ', relation_rms)

#%% 

## Autocorrelation


## Cross-correlation
#multimodal_corr = np.correlate(ECG_peaks, PCG_peaks, mode='full')
multimodal_corr = cross_corr(ECG_peaks, PCG_peaks)
# plt.figure()
# plt.plot(multimodal_corr)
# plt.grid()

## Synchronize PCG with ECG
corr_idx = (np.where(multimodal_corr == np.max(multimodal_corr))[0][0])
PCG_peaks = PCG_peaks[corr_idx:]
PCG_peaks_lagged = np.pad(PCG_peaks, (0,corr_idx), 'constant', constant_values=0)

# plt.figure()
# plt.plot(ECG_peaks) 
# plt.plot(PCG_peaks_lagged)
# plt.grid()


## ECG - lagged PCG
rest = ECG_peaks - PCG_peaks_lagged


ECG_noise = np.where(rest > 0, rest, 0)
S1_noise = np.where(rest < 0, rest, 0)

# visualizeStacked(ECG_noise, S1_noise)

## Clean S2

S2_clean = PCG_peaks_lagged + S1_noise
# plt.figure()
# plt.plot(S2_clean) 
# plt.grid()

## get lag between clean s2 with inverted s1 noise
#S2_S1_corr = np.correlate(S2_clean, -S1_noise, mode='full')
S2_S1_corr = cross_corr_back(S2_clean, -S1_noise)
S2S1corr_idx = (S2_S1_corr.size - (np.where(S2_S1_corr == np.max(S2_S1_corr))[0][0]))

S1_noise_lag = np.pad(-S1_noise, (S2S1corr_idx,0), 'constant', constant_values=0)
S1_noise_lag = S1_noise_lag[: - (S2S1corr_idx)]
  
# plt.figure()
# plt.plot(S2_clean)
# plt.plot(S1_noise_lag)
# plt.grid()   

## PCG noise Clean S2 - S1_noise_lag
PCG_noise = S2_clean - S1_noise_lag              
# plt.figure()
# plt.plot(PCG_noise)
# plt.grid() 

## Calculate SNRs

SNRecg = RMS(ECG_peaks)/(RMS(ECG_noise))
SNRpcg = RMS(PCG_peaks)/(RMS(PCG_noise))

mSQI = harmonic_mean(SNRecg, SNRpcg)

print('\n SNR ECG = ', SNRecg, ' SNR PCG = ', SNRpcg, ' mSQI = ', mSQI)













