# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:16:31 2024

@author: danie
"""

# %%
# Imports

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import pywt
import pydub
import time
import wfdb
import sounddevice as sd
from scipy import signal
import preprocessing_lib as pplib
import feature_extraction_lib as ftelib
import file_process_lib as importlib

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


import copy

# %%


# %% Denoising
# Import and analyze the dataset

directory = r'../LUDB/data/2'

# Read as record
record = wfdb.rdrecord(directory)
wfdb.plot_wfdb(record=record, title="Record 1 from LUDB")

# Read only signals
signals, fields = wfdb.rdsamp(directory, channels=[1, 3, 5, 6])

# Read annotations
ann = wfdb.rdann(directory, extension="i")
wfdb.plot_wfdb(annotation=ann)

# Plot annotations on top of signal
wfdb.plot_wfdb(record=record, annotation=ann,
               title="I lead annotated (not differentiated)")

# indices where the annotation is applied
annotation_index = ann.sample

# symbol order of the annotations
annotation_vector = ann.symbol
