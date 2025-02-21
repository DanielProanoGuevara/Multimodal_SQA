# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:30:56 2025

@author: danie
"""
import os
import pandas as pd
import wfdb
import numpy as np

base_directory = r'../LUDB/data/'

compiled_data = []

for n in range(1, 201):
    directory = os.path.join(base_directory, str(n))

    try:
        ann = wfdb.rdann(directory, 'i')
        ann_idx = ann.sample
        ann_lbl = ann.symbol

        signals, fields = wfdb.rdsamp(directory)
        sig_name = fields['sig_name']

        m = signals.shape[0]
        labels = ['x'] * m

        for i in range(len(ann_idx)):
            if ann_lbl[i] == '(' and i + 2 < len(ann_idx) and ann_lbl[i+2] == ')':
                start = ann_idx[i]
                end = ann_idx[i+2]
                central_label = ann_lbl[i+1]
                labels[start:end+1] = [central_label] * (end - start + 1)

        sampfrom = ann_idx[0]
        sampto = ann_idx[-1]
        cropped_signals = signals[sampfrom:sampto + 1, :]
        cropped_labels = labels[sampfrom:sampto + 1]

        label_data = cropped_labels

        for i, signal_name in enumerate(sig_name):
            signal_data = cropped_signals[:, i].tolist()
            compiled_data.append({
                'id': f'{n}_{signal_name}',
                'signal': signal_data,
                'label': label_data
            })
    except Exception as e:
        print(f"Error processing sample {n}: {e}")

final_dataframe = pd.DataFrame(compiled_data)


# Save the DataFrames to pickle files
ludb_path = r'..\ludb_full.pkl'
final_dataframe.to_pickle(ludb_path)
