# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:22:58 2025

@author: danie
"""

import numpy as np
import pandas as pd

import preprocessing_lib as pplib
import feature_extraction_lib as ftelib


def process_dataframe(df):
    """
    Processes the input dataframe and returns two separate dataframes for ECG and PCG signals.

    Parameters:
    df (pd.DataFrame): Original dataframe containing 'ID', 'Auscultation Point', 'ECG Signal', and 'PCG Signal'.
    downsample (function): Private function to downsample the PCG signal.

    Returns:
    df_ecg (pd.DataFrame): Processed ECG dataframe with z-score standardization.
    df_pcg (pd.DataFrame): Processed PCG dataframe with downsampling and z-score standardization.
    """

    # Separate ECG and PCG data
    df_ecg = df[['ID', 'Auscultation Point', 'ECG Signal']].rename(
        columns={'ECG Signal': 'Signal'})
    df_pcg = df[['ID', 'Auscultation Point', 'PCG Signal']].rename(
        columns={'PCG Signal': 'Signal'})

    # Remove empty lists
    df_ecg = df_ecg[df_ecg['Signal'].apply(lambda x: len(x) > 0)]
    df_pcg = df_pcg[df_pcg['Signal'].apply(lambda x: len(x) > 0)]

    # Apply downsampling only for IDs 1 to 108 (index 0 to 404)
    def downsample_pcg(signal, idx):
        if idx <= 404:  # IDs 1 to 108 correspond to index range 0 to 404
            return pplib.downsample(signal, orig_freq=8000, target_freq=3000)
        return signal  # Keep the rest unchanged

    df_pcg['Signal'] = [downsample_pcg(sig, i)
                        for i, sig in enumerate(df_pcg['Signal'])]

    # Apply z-score standardization
    df_ecg['Signal'] = df_ecg['Signal'].apply(
        lambda x: pplib.z_score_standardization(x) if len(x) > 1 else x)
    df_pcg['Signal'] = df_pcg['Signal'].apply(
        lambda x: pplib.z_score_standardization(x) if len(x) > 1 else x)

    return df_ecg, df_pcg


root_dir = r'..\ECG_PCG_structured_data.pkl'
df = pd.read_pickle(root_dir)

ecg_df, pcg_df = process_dataframe(df)


ecg_pickle_path = r'..\DatasetCHVNGE\ecg_ulsge.pkl'
pcg_pickle_path = r'..\DatasetCHVNGE\pcg_ulsge.pkl'

ecg_df.to_pickle(ecg_pickle_path)
print(f"ECG DataFrame saved to {ecg_pickle_path}")

pcg_df.to_pickle(pcg_pickle_path)
print(f"PCG DataFrame saved to {pcg_pickle_path}")
