# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:32:55 2024

@author: danie
"""

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

# U-NET architecture


def unet_pcg(nch, patch_size, dropout=0.05):
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


# %% Import Files
# Load the train and validation datasets
val_df = pd.read_pickle(r'..\features_signals_wv.pkl')

val_df['Homomorphic'] = val_df['Features'].apply(lambda x: x[:, 0])
val_df['CWT_Morl'] = val_df['Features'].apply(lambda x: x[:, 1])
val_df['CWT_Mexh'] = val_df['Features'].apply(lambda x: x[:, 2])
val_df['Hilbert_Env'] = val_df['Features'].apply(lambda x: x[:, 3])
val_df = val_df.drop(columns=['Features'])

# Convert the loaded DataFrames to numpy arrays
val_data = val_df[['ID', 'Homomorphic', 'CWT_Morl',
                   'CWT_Mexh', 'Hilbert_Env']].to_numpy()

# Feature creation
BATCH_SIZE = 4
patch_size = 64
nch = 4
stride = 8

# Create patches and structures for NN training
val_features = ftelib.process_dataset_no_labels(val_data, patch_size, stride)

# %% Neural Network model

checkpoint_path = '../pcg_unet_weights/checkpoint_wv.keras'

EPOCHS = 15
learning_rate = 1e-4
model = unet_pcg(nch, patch_size=patch_size)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
              metrics=['CategoricalAccuracy', 'Precision', 'Recall'])


# %% Inference pipeline

model.load_weights(checkpoint_path)
val_test = model.predict(val_features)

# Reconstruct from patches

# Get original lengths from validation data
original_lengths = [len(seq) for seq in val_data[:, 1]]
reconstructed_labels = ftelib.reconstruct_original_data(
    val_test, original_lengths, patch_size, stride)

# %% Save Probabilities

predictions_pickle_path = r'..\ULSGE_pred_wv.pkl'

with open(predictions_pickle_path, 'wb') as file:
    pickle.dump(reconstructed_labels, file)
