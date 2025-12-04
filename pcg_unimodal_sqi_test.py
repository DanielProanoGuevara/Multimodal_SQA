# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 15:15:20 2025

@author: Asus
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing_lib as pplib
import feature_extraction_lib as ftelib
import file_process_lib as importlib
import sqi_pcg_lib
from sklearn.preprocessing import OneHotEncoder

import pickle
import copy

# %% Import PCG
root_dir = r'..\DatasetCHVNGE\pcg_ulsge.pkl'
pcg_df_original = pd.read_pickle(root_dir)
# Deep copy
pcg_df = copy.deepcopy(pcg_df_original)

# %% Preprocess PCG

pcg_df['PCG'] = pcg_df['PCG'].apply(
    lambda data: pplib.butterworth_filter(
        data,
        filter_topology = 'bandpass', 
        order = 4, 
        fs = 3000, 
        fc = [50, 250]
    )
)

# %% Feature extraction
## Test score generation

seSQI = sqi_pcg_lib.se_sqi_pcg(pcg_df['PCG'][0], 3000, M=2, r=0.0008)
