#%% Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import pydub
import re
import pywt

#%% Functions
def merge_rows(group):
    # Initialize merged_row with the first row in the group
    merged_row = group.iloc[0].copy()
    # Iterate over columns
    for col in group.columns[2:]:
        # If there is a non-null value in the current column, update merged_row
        if not group[col].isnull().all():
            merged_row[col] = group[col].dropna().iloc[0]
    return merged_row

def softThreshold (signal, dev):
    lbda = dev * np.sqrt(2 * np.log(len(signal))) # Universal threshold calculator
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

def residuals (filtered, raw):
    return  raw - filtered[:len(raw)]

def ecgWVDenoise(signal, family, level, threshold):
    wv_coeffs = pywt.wavedec(signal, wavelet=family, level=level)
    sigma = np.median(abs(wv_coeffs[-2]))/0.674
    for i in range(len(wv_coeffs)-1):
        lbda = threshold(wv_coeffs[i+1], sigma)
        wv_coeffs[i+1] = softThreshold (wv_coeffs[i+1], lbda)
    wv_coeffs[0][:] = 0
    return pywt.waverec(wv_coeffs, wavelet=family)

def pcgWVDenoise(signal, threshold):
    wv_coeffs = pywt.wavedec(signal, wavelet='coif4', level=5)
    sigma = np.median(abs(wv_coeffs[1]))/0.674
    for i in range(len(wv_coeffs)):
        lbda = threshold(wv_coeffs[i], sigma)
        wv_coeffs[i] = softThreshold (wv_coeffs[i], lbda)
    wv_coeffs[0] -= np.mean(wv_coeffs[0])
    wv_coeffs[-1][:] = 0
    wv_coeffs[-2][:] = 0
    wv_coeffs[-3][:] = 0
    return pywt.waverec(wv_coeffs, wavelet='coif4')


def totalPNID(noise, raw):
    noise_power = signalPower(noise)
    raw_power = signalPower(raw)
    return 10 * np.log10(noise_power / raw_power)

def meanSquareError(filtered, raw):
    filtered = np.array(filtered)
    raw = np.array(raw)
    return np.mean((raw - filtered[:len(raw)]) ** 2)

def signalPower(signal):
    return np.mean(signal **2)

def SNR(signal, noise):
    return 10*np.log10(signalPower(signal)/signalPower(noise))


def denoiseECG(ECG):
    ## Wavelet denoising
    ECG_wv_denoised = ecgWVDenoise(ECG, 'db4', 8, universal)
    return ECG_wv_denoised

def denoisePCG(PCG):
    ## Wavelet Denoise
    PCG_wv_denoised = pcgWVDenoise(PCG, normalshrink_lambda)
    return PCG_wv_denoised



#%% Process dataset
# Define the sample directory
directory_dataset = 'DatasetCHVNGE'
paths = []

data = []

#Constants

ECG_rate = 500
ECG_bit_width = 12
ECG_resolution = (2 ** ECG_bit_width)-1
PCG_bit_width = 16
PCG_resolution = (2 ** PCG_bit_width)-1


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
            ## Normalize full-scale
            PCG_v = (PCG ) / (PCG_resolution) #uint 16 bits (scale -0.5;0.5)
            
            ECG_rate = None
            ECG = None
        elif filename.endswith(".raw"):
            ECG = np.loadtxt(filename, delimiter=",", dtype=int)
            ## Normalize full-scale
            ECG_v = (ECG - ECG_resolution/2) / (ECG_resolution) #int 12 bits (scale -0.5;0.5)
            
            PCG = None
            PCG_rate = None
            # Apend data to the list
            data.append([patient, auscultation_point, ECG_v, PCG_v])
        
# Create a DataFrame from collectedd data
db_df = pd.DataFrame(data, columns=["Patient", "Auscultation_Point", "ECG", "PCG"])
df_merged = db_df.groupby(["Patient", "Auscultation_Point"]).apply(merge_rows)
df_merged.reset_index(drop=True,inplace = True)

#%% Denoise ECG and PCG
df_merged['ECG_Denoised'] = df_merged['ECG'].apply(denoiseECG)
df_merged['PCG_Denoised'] = df_merged['PCG'].apply(denoisePCG)
#df_merged.to_pickle("./Multimodal_SQA/processedDataBase.pkl")
# %% Residuals and SNR metrics
df_merged['Residuals_ECG'] = df_merged.apply(lambda x: residuals(x['ECG_Denoised'], x['ECG']), axis=1)
df_merged['Residuals_PCG'] = df_merged.apply(lambda x: residuals(x['PCG_Denoised'], x['PCG']), axis=1)

# MSE
df_merged['MSE_ECG'] = df_merged['Residuals_ECG'].apply(signalPower)
df_merged['MSE_PCG'] = df_merged['Residuals_PCG'].apply(signalPower)

# ECG SNR
df_merged['SNR_ECG'] = df_merged.apply(lambda x: totalPNID(x['Residuals_ECG'], x['ECG']), axis=1)

# PCG SNR
df_merged['SNR_PCG'] = df_merged.apply(lambda x: SNR(x['PCG_Denoised'], x['Residuals_PCG']), axis=1)


#print(df_merged.head())

#%% Import manual annotations

manualQScores = pd.read_excel('DatasetCHVNGE/SignalQuality.xlsx', 
                              sheet_name='Total')

manualQScores = manualQScores.sort_values(by=['Trial', 'Spot'])
manualQScores.reset_index(drop=True, inplace=True)

# Eliminate the missing data values
filteredManualQ = manualQScores[~manualQScores['Trial'].between(71, 77)]
filteredManualQ.reset_index(drop=True, inplace=True)

#print(manualQScores.head())


#%% Plots

# ECG MSE
plt.figure()
plt.scatter(filteredManualQ['MINMAX_ECG'], df_merged['MSE_ECG'])
plt.grid()
plt.xlabel('Quality')
plt.ylabel('MSE')
plt.title('ECG Mean Square Error quality comparison')
plt.show()

# ECG SNR
plt.figure()
plt.scatter(filteredManualQ['MINMAX_ECG'], df_merged['SNR_ECG'])
plt.grid()
plt.xlabel('Quality')
plt.ylabel('Noise Distortion')
plt.title('ECG Noise distortion quality comparison')
plt.show()

# PCG MSE
plt.figure()
plt.scatter(filteredManualQ['MINMAX_PCG'], df_merged['MSE_PCG'])
plt.grid()
plt.xlabel('Quality')
plt.ylabel('MSE')
plt.title('PCG Mean Square Error quality comparison')
plt.show()

# ECG SNR
plt.figure()
plt.scatter(filteredManualQ['MINMAX_PCG'], df_merged['SNR_PCG'])
plt.grid()
plt.xlabel('Quality')
plt.ylabel('SNR')
plt.title('PCG SNR vs Quality')
plt.show()
