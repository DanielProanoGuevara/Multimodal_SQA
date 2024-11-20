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
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, UpSampling1D, concatenate
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import preprocessing_lib as pplib
import feature_extraction_lib as ftelib
import file_process_lib as importlib

import copy

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


def reconstruct_original_data(patched_data, original_lengths, patch_size, stride):
    """
    Reconstruct the original sequences from patched data.

    Parameters
    ----------
    patched_data : numpy.ndarray
        Patched data array of shape (Num_Patches, Patch_Size, Num_Features).
    original_lengths : list
        List of original lengths for each patient sequence.
    patch_size : int
        The number of samples for each patch.
    stride : int
        The number of samples to stride between patches.

    Returns
    -------
    reconstructed_data : list of numpy.ndarray
        List containing the reconstructed data for each patient.
    """
    reconstructed_data = []
    current_idx = 0

    for original_length in original_lengths:
        # Initialize arrays to hold the reconstructed sequence and overlap count
        reconstructed = np.zeros((original_length, patched_data.shape[-1]))
        overlap_count = np.zeros(original_length)

        num_patches = int(
            np.floor((original_length - patch_size) / stride)) + 1
        adjusted_stride_samples = (
            (original_length - patch_size) / (num_patches - 1)
            if num_patches > 1 else stride
        )
        adjusted_stride_samples = int(round(adjusted_stride_samples))

        # Iterate over patches and reconstruct the sequence
        for i in range(num_patches):
            start_idx = i * adjusted_stride_samples
            end_idx = min(start_idx + patch_size, original_length)

            reconstructed[start_idx:end_idx] += patched_data[current_idx,
                                                             :end_idx - start_idx, :]
            overlap_count[start_idx:end_idx] += 1

            current_idx += 1

        # Average the overlapping regions
        reconstructed /= np.maximum(overlap_count[:, None], 1)
        reconstructed_data.append(reconstructed)

    return reconstructed_data


def reverse_one_hot_encoding(one_hot_encoded_data, desired_order=[0, 1, 2, 3]):
    """
    Reverse the one-hot encoding to get the original labels.

    Parameters
    ----------
    one_hot_encoded_data : numpy.ndarray
        One-hot encoded data of shape (Num_Samples, Num_Classes).
    desired_order : list
        List representing the label order used during one-hot encoding.

    Returns
    -------
    labels : numpy.ndarray
        Array of decoded labels of shape (Num_Samples,).
    """
    # Use argmax to find the index of the maximum value in each one-hot encoded row
    label_indices = np.argmax(one_hot_encoded_data, axis=1)
    labels = np.array([desired_order[idx] for idx in label_indices])
    return labels


def max_temporal_modelling(seq, num_states=4):
    for t in range(1, len(seq)):
        if seq[t] != seq[t-1] and seq[t] != ((seq[t-1] + 1) % num_states):
            seq[t] = seq[t-1]
    return seq


def unet_pcg(nch, patch_size, dropout=0.0):
    inputs = tf.keras.layers.Input(shape=(patch_size, nch))
    conv1 = tf.keras.layers.Conv1D(
        8, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv1D(
        8, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)
    pool1 = tf.keras.layers.Dropout(dropout)(pool1)

    conv2 = tf.keras.layers.Conv1D(
        16, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv1D(
        16, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2)
    pool2 = tf.keras.layers.Dropout(dropout)(pool2)

    conv3 = tf.keras.layers.Conv1D(
        32, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv1D(
        32, 3, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv3)
    pool3 = tf.keras.layers.Dropout(dropout)(pool3)

    conv4 = tf.keras.layers.Conv1D(
        64, 3, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv1D(
        64, 3, activation='relu', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv4)
    pool4 = tf.keras.layers.Dropout(dropout)(pool4)

    conv5 = tf.keras.layers.Conv1D(
        128, 3, activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.Conv1D(
        128, 3, activation='relu', padding='same')(conv5)

    up6_prep = tf.keras.layers.UpSampling1D(size=2)(conv5)

    up6 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv1D(64, 2, padding='same')(up6_prep), conv4], axis=2)
    up6 = tf.keras.layers.Dropout(dropout)(up6)
    conv6 = tf.keras.layers.Conv1D(
        64, 3, activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.Conv1D(
        64, 3, activation='relu', padding='same')(conv6)

    up7_prep = tf.keras.layers.UpSampling1D(size=2)(conv6)

    up7 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv1D(64, 2, padding='same')(up7_prep), conv3], axis=2)
    up7 = tf.keras.layers.Dropout(dropout)(up7)
    conv7 = tf.keras.layers.Conv1D(
        32, 3, activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.Conv1D(
        32, 3, activation='relu', padding='same')(conv7)

    up8_prep = tf.keras.layers.UpSampling1D(size=2)(conv7)

    up8 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv1D(32, 2, padding='same')(up8_prep), conv2], axis=2)
    up8 = tf.keras.layers.Dropout(dropout)(up8)
    conv8 = tf.keras.layers.Conv1D(
        16, 3, activation='relu', padding='same')(up8)
    conv8 = tf.keras.layers.Conv1D(
        16, 3, activation='relu', padding='same')(conv8)

    up9_prep = tf.keras.layers.UpSampling1D(size=2)(conv8)

    up9 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv1D(8, 2, padding='same')(up9_prep), conv1], axis=2)
    up9 = tf.keras.layers.Dropout(dropout)(up9)
    conv9 = tf.keras.layers.Conv1D(
        8, 3, activation='relu', padding='same')(up9)
    conv9 = tf.keras.layers.Conv1D(
        8, 3, activation='tanh', padding='same')(conv9)

    conv10 = tf.keras.layers.Conv1D(4, 1, activation='softmax')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[conv10])
    return model


# %% Simplify PCG notebook

# Load the train and validation datasets
train_df = pd.read_pickle('../train_physionet_2016.pkl')
val_df = pd.read_pickle('../validation_physionet_2016.pkl')

# Convert the loaded DataFrames to numpy arrays
train_data = train_df[['Patient ID', 'Homomorphic',
                       'CWT_Morl', 'CWT_Mexh', 'Hilbert_Env', 'Labels']].to_numpy()
val_data = val_df[['Patient ID', 'Homomorphic', 'CWT_Morl',
                   'CWT_Mexh', 'Hilbert_Env', 'Labels']].to_numpy()

# Feature creation
BATCH_SIZE = 4
patch_size = 64
nch = 4
stride = 8


train_features, train_labels = process_dataset(train_data, patch_size, stride)
val_features, val_labels = process_dataset(val_data, patch_size, stride)

# %% NN

checkpoint_path = '../pcg_unet_weights/checkpoint.keras'

EPOCHS = 15
learning_rate = 1e-4
model = unet_pcg(nch, patch_size=patch_size)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
              metrics=['CategoricalAccuracy', 'Precision', 'Recall'])

model.load_weights(checkpoint_path)

# %% Inference

predictions_train = model.predict(train_features)
val_test = model.predict(val_features)

# %% Reconstruct from patches

# Get original lengths from validation data
original_lengths = [len(seq) for seq in val_data[:, 1]]
reconstructed_labels = reconstruct_original_data(
    val_test, original_lengths, patch_size, stride)

# %% Reverse one-hot encoding
pred_labels = [reverse_one_hot_encoding(pred) for pred in reconstructed_labels]

prediction_labels = copy.deepcopy(pred_labels)

ground_truth = [reverse_one_hot_encoding(pred) for pred in val_data[:, 5]]

predictions = np.array([max_temporal_modelling(prediction)
                       for prediction in prediction_labels], dtype=object)


# %%
plt.plot(pred_labels[10])
plt.plot(predictions[10])
plt.plot(ground_truth[10])
