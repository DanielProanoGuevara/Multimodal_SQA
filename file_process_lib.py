"""
File Processing Library.

@file: file_process_lib.py

@coding: utf_8

@description: Contains the importing datasets functions. Contains the training,
test, and validation set splitter, contains the k-fold splitter, contains the
database processor.

@functions:
-


@author: Daniel Proaño-Guevara.

@creationDate: 2024-09-26

@version: 0.1
"""

import os
import numpy as np
import scipy.io
from scipy.io import wavfile


def import_physionet_2016(signals_path, labels_path):
    # Import audio file
    samplerate, original_data = wavfile.read(signals_path)

    # Import labels
    raw_pre = scipy.io.loadmat(labels_path)
    raw_label = raw_pre.get('state_ans0', [])
    # Format the output
    formatted_output = [[int(item[0].item()), str(
        item[1].item()).strip("[]'")] for item in raw_label]
    # Convert into NumPy array
    formatted_array = np.array(formatted_output)

    # Create an empty array for the propagated labels
    propagated_labels = np.empty(len(original_data), dtype=object)
    # Extract the indices and labels
    # Subtract 1 to convert MATLAB indices to Python
    indices = formatted_array[:, 0].astype(int) - 1
    labels = formatted_array[:, 1]
    # Propagate labels between the indices
    for i in range(len(indices)):
        start = indices[i]
        end = indices[i+1] if i+1 < len(indices) else len(original_data)
        propagated_labels[start:end] = labels[i]

    return [samplerate, original_data, propagated_labels]
