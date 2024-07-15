# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:16:31 2024

@author: danie
"""

#%% 
# Imports

import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.stats import norm


#%% 
# Throwaway Functions

def load_wav_files(directory):
    # List to store the data from all .wav files
    data_list = []
    
    # Loop through all files in the given directory
    for filename in os.listdir(directory):
        # Check if the file is a .wav file
        if filename.endswith('.wav'):
            # Get the full path to the file
            file_path = os.path.join(directory, filename)
            # Read the .wav file
            samplerate, data = wavfile.read(file_path)
            # Append the data to the list
            data_list.append(data)
    
    # Combine all data into a single numpy array
    combined_data = np.concatenate(data_list)
    
    return combined_data

def plot_histogram_with_gaussian(data):
    # Calculate histogram
    counts, bins, _ = plt.hist(data, bins=50, density=True, alpha=0.6, color='g')
    
    # Fit a Gaussian distribution to the data
    mu, std = norm.fit(data)
    
    # Plot the Gaussian fit
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    
    title = f"Fit results: mu = {mu:.2f},  std = {std:.2f}"
    plt.title(title)
    
    plt.show()
    
    return mu, std

#%%
# Import and analyze the dataset

# Directory containing the files
directory = '../Physionet_2016_training/training-f'

# Load and combine .wav files
combined_wav_data = load_wav_files(directory)

# Print the shape of the combined array
print(combined_wav_data.shape)

#%%
# Generate histogram and fit Gaussian
mean, std_dev = plot_histogram_with_gaussian(combined_wav_data)

# Calculate additional statistics
data_min = np.min(combined_wav_data)
data_max = np.max(combined_wav_data)

# Print the statistics
print(f"Mean: {mean}")
print(f"Standard Deviation: {std_dev}")
print(f"Minimum: {data_min}")
print(f"Maximum: {data_max}")