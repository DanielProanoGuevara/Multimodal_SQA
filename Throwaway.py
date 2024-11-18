# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:16:31 2024

@author: danie
"""

# %%
# Imports

# import os
# import glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# import pywt
# import pydub
# import time
# import wfdb
# import sounddevice as sd
# from scipy import signal
# import preprocessing_lib as pplib
# import feature_extraction_lib as ftelib
# import file_process_lib as importlib

# import os

# import librosa
# import logging
# import numpy as np
# import pandas as pd
# import scipy.io as sio
# import scipy.signal
# import re

# import pickle

# from scipy.io import wavfile
# import tensorflow as tf
# # from tqdm import tqdm
# import matplotlib.pyplot as plt

import os

import librosa
import logging
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal
import re

import pickle

from scipy.io import wavfile
import tensorflow as tf
import matplotlib.pyplot as plt

import preprocessing_lib as pplib
import feature_extraction_lib as ftelib
import file_process_lib as importlib

# %%


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
# %% Functions

# Process the entire dataset to create patches
def process_dataset(data, patch_size, stride):
    all_features = []
    all_labels = []
    for i in range(data.shape[0]):
        features = np.stack(data[i, 1:5], axis=-1)
        labels = data[i, 5]
        features_patches, labels_patches = ftelib.create_patches(
            features, labels, patch_size, stride)
        all_features.append(features_patches)
        all_labels.append(labels_patches)
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_features, all_labels


def reconstruct_patches(predictions, original_length, patch_size, stride):
    reconstructed = np.zeros((original_length, predictions.shape[1]))
    overlap_count = np.zeros(original_length)
    num_patches = predictions.shape[2]
    for i in range(num_patches):
        start_idx = i * stride
        end_idx = min(start_idx + patch_size, original_length)
        reconstructed[start_idx:end_idx] += predictions[:,
                                                        :, i].T[:end_idx - start_idx]
        overlap_count[start_idx:end_idx] += 1
    reconstructed /= np.maximum(overlap_count[:, None], 1)
    return reconstructed


# %% Simplify PCG notebook

# Load the train and validation datasets
train_df = pd.read_pickle('../train_physionet_2016.pkl')
val_df = pd.read_pickle('../validation_physionet_2016.pkl')

# Convert the loaded DataFrames to numpy arrays
train_data = train_df[['Patient ID', 'Homomorphic',
                       'CWT_Morl', 'CWT_Mexh', 'Hilbert_Env', 'Labels']].to_numpy()
val_data = val_df[['Patient ID', 'Homomorphic', 'CWT_Morl',
                   'CWT_Mexh', 'Hilbert_Env', 'Labels']].to_numpy()

# Feature creatino
patch_size = 64
stride = 8

train_features, train_labels = process_dataset(train_data, patch_size, stride)
val_features, val_labels = process_dataset(val_data, patch_size, stride)

# reconstructed_train = [reconstruct_patches(pred, len(
#     train_data[i][2]), patch_size, stride) for i, pred in enumerate(train_labels)]
