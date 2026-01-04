# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 15:11:58 2025

@author: danie
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing_lib as pplib
import feature_extraction_lib as ftelib
import file_process_lib as importlib
import sqi_ecg_lib
from sklearn.preprocessing import OneHotEncoder

import pickle
import copy

# %% Import ECG
root_dir = r'..\DatasetCHVNGE\ecg_ulsge.pkl'
ecg_df_original = pd.read_pickle(root_dir)
# Deep copy
ecg_df = copy.deepcopy(ecg_df_original)

# %% Preprocess PCG

ecg_df['ECG'] = ecg_df['ECG'].apply(
    lambda data: pplib.butterworth_filter(
        data,
        filter_topology='lowpass',
        order=4,
        fs=500,
        fc=125
    )
)

# %% Feature extraction
# Test score generation

N=8

plt.plot(ecg_df['ECG'][N])

bsqi = sqi_ecg_lib.bSQI(ecg_df['ECG'][N], 500)

psqi = sqi_ecg_lib.pSQI(ecg_df['ECG'][N], 500)

ssqi = sqi_ecg_lib.sSQI(ecg_df['ECG'][N])

ksqi = sqi_ecg_lib.kSQI(ecg_df['ECG'][N])

fsqi = sqi_ecg_lib.fSQI(ecg_df['ECG'][N], 500)

bassqi = sqi_ecg_lib.basSQI(ecg_df['ECG'][N], 500)
