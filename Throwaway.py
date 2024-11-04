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
# from tqdm import tqdm
import matplotlib.pyplot as plt

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
# %% Opening dataset training Miguel
# with open('../train_physionet_2016.pkl', 'rb') as f:
#     data = pickle.load(f)

train = np.load('../train_data.npy', allow_pickle=True)
val = np.load('../val_data.npy', allow_pickle=True)
test = np.load('../test_data.npy', allow_pickle=True)


def filter_smaller_than_patch(features, patch_size):
    # Remove sounds shorter than patch size and return their indices
    return np.array([j for j in range(len(features)) if len(features[j]) >= patch_size], dtype=int)


patch_size = 64
nch = 2
stride = 32

# Ensure indices are integers and apply them correctly to filter the datasets
train_indices = filter_smaller_than_patch(train[:, 2], patch_size)
val_indices = filter_smaller_than_patch(val[:, 2], patch_size)
test_indices = filter_smaller_than_patch(test[:, 2], patch_size)

train = train[train_indices, ...]
val = val[val_indices, ...]
test = test[test_indices, ...]


def one_hot(label_column: np.array, num_states: int = 4):
    # your code here
    label_column = label_column - 1  # make it go to [0, 1, 2, 3]
    return np.eye(num_states)[label_column]


def apply_one_hot(labels): return np.array(
    [one_hot(label) for label in labels], dtype=object)


train[:, 5] = apply_one_hot(train[:, 5])
val[:, 5] = apply_one_hot(val[:, 5])
test[:, 5] = apply_one_hot(test[:, 5])


class PCGDataPreparer:
    def __init__(self, patch_size: int, stride: int, number_channels: int = 2, num_states: int = 4):
        self.patch_size = patch_size
        self.stride = stride
        self.number_channels = number_channels
        self.num_states = num_states
        self.features = None
        self.labels = None

    def _compute_pcg_patches(self, sound, label):
        # TODO: ask them to implement this
        num_samples = len(sound)
        # TODO: they should complete this for
        num_windows = int((num_samples - self.patch_size) / self.stride) + 1
        for window_idx in range(num_windows):
            patch_start = window_idx * self.stride
            yield sound[patch_start:patch_start + self.patch_size, :],  label[patch_start: patch_start + self.patch_size, :]

        window_remain = num_samples - self.patch_size
        if window_remain % self.stride > 0:
            yield sound[window_remain:, :], label[window_remain:, :]

    def set_features_and_labels(self, features, labels):
        self.features = features
        self.labels = labels
        num_observations = len(self.features)
        total_windows = 0
        for obs in features:
            num_samples = len(features)
            num_windows = int(
                (num_samples - self.patch_size) / self.stride) + 1
            window_remain = num_samples - self.patch_size
            if window_remain % self.stride > 0:
                num_windows += 1
            total_windows += num_windows
        self.num_steps = total_windows

    def __call__(self):
        num_observations = len(self.labels)
        for obs_idx in range(num_observations):
            # np.column_stack
            features = tf.stack(self.features[obs_idx], axis=1)
            labels = self.labels[obs_idx]
            for s, y in (self._compute_pcg_patches(features, labels)):
                yield s, y


patch_size = 64
nch = 2
stride = 32
train_dp = PCGDataPreparer(patch_size=patch_size,
                           number_channels=nch,
                           stride=stride,
                           num_states=4)
train_dp.set_features_and_labels(train[:, [3, 4]], train[:, 5])

val_dp = PCGDataPreparer(patch_size=patch_size,
                         number_channels=nch,
                         stride=stride,
                         num_states=4)
val_dp.set_features_and_labels(val[:, [3, 4]], val[:, 5])

test_dp = PCGDataPreparer(patch_size=patch_size,
                          number_channels=nch,
                          stride=stride,
                          num_states=4)
test_dp.set_features_and_labels(test[:, [3, 4]], test[:, 5])
