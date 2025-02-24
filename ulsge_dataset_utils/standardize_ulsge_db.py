# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 2025

@author: Daniel Proaño-Guevara
"""

import sys
import os
import numpy as np
import pandas as pd

# Get the absolute path of the mother folder
origi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add origi folder to sys.path
sys.path.append(origi_path)

import preprocessing_lib as pplib


def process_signal_data(pkl_file):
    """
    Processes a DataFrame containing signal data stored in a pickle (.pkl) file.
    
    Input DataFrame Columns:
        - 'ID'
        - 'Auscultation_Point'
        - 'Source'
        - 'ECG'
        - 'PCG'
    
    Outputs:
        1. ECG DataFrame:
           - Retains only 'ID', 'Auscultation_Point', and 'ECG'
           - Applies z-score standardization to the 'ECG' column
           - Removes rows where 'ECG' is an empty list
           
        2. PCG DataFrame:
           - Retains only 'ID', 'Auscultation_Point', and 'PCG'
           - For rows with 'Source' == "Rijuven", decimates the signal (8 kHz → 3 kHz) using downsample()
           - Applies z-score standardization to the 'PCG' column
           - Removes rows where 'PCG' is an empty list
           
    Parameters:
        pkl_file (str): Path to the pickle file containing the DataFrame.
        
    Returns:
        tuple: (df_ecg, df_pcg) where:
            - df_ecg is the processed ECG DataFrame.
            - df_pcg is the processed PCG DataFrame.
    """
    try:
        df = pd.read_pickle(pkl_file)
    except Exception as e:
        raise ValueError(f"Error loading pickle file: {e}")
    
    # Ensure all required columns exist
    required_cols = ['ID', 'Auscultation_Point', 'Source', 'ECG', 'PCG']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    #############################################
    # Process ECG DataFrame
    #############################################
    # Select only the relevant columns
    df_ecg = df[['ID', 'Auscultation_Point', 'ECG']].copy()
    
    # Remove rows where 'ECG' is not a non-empty list
    df_ecg = df_ecg[df_ecg['ECG'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    
    # Apply z-score standardization to the 'ECG' column.
    # This uses the externally provided z_score_standardization() function.
    df_ecg['ECG'] = df_ecg['ECG'].apply(lambda signal: pplib.z_score_standardization(signal))
    
    #############################################
    # Process PCG DataFrame
    #############################################
    # Select the columns required for PCG processing (include 'Source' for conditional processing)
    df_pcg = df[['ID', 'Auscultation_Point', 'PCG', 'Source']].copy()
    
    # Remove rows where 'PCG' is not a non-empty list
    df_pcg = df_pcg[df_pcg['PCG'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    
    def process_pcg_signal(row):
        """
        Processes a single row's PCG signal:
          - If 'Source' is "Rijuven", downsample the signal.
          - Then, apply z-score standardization.
          - Returns None if processing fails.
        """
        signal = row['PCG']
        try:
            if row['Source'] == "Rijuven":
                # Downsample the signal from 8 kHz to 3 kHz using the provided function.
                signal = pplib.downsample(signal, 8000, 3000)
            # Apply z-score standardization using the provided z_score_standardization() function.
            return pplib.z_score_standardization(signal)
        except Exception:
            # If any error occurs (e.g., malformed data), return None.
            return None

    # Apply the processing function row-wise.
    df_pcg['PCG'] = df_pcg.apply(process_pcg_signal, axis=1)
    
    # Remove any rows where PCG processing failed (i.e., resulted in None)
    df_pcg = df_pcg[df_pcg['PCG'].notnull()]
    
    # Drop the 'Source' column as it is no longer needed.
    df_pcg = df_pcg.drop(columns=['Source'])
    
    return df_ecg, df_pcg

# Example usage:
if __name__ == "__main__":
    pkl_file = "../DatasetCHVNGE/compiled_dataset.pkl"  # Path to your pickle file
    ecg_pickle_path = "../DatasetCHVNGE/ecg_ulsge.pkl"
    pcg_pickle_path = "../DatasetCHVNGE/pcg_ulsge.pkl"
    try:
        ecg_df, pcg_df = process_signal_data(pkl_file)
        print("Processed ECG DataFrame:")
        print(ecg_df.head())
        ecg_df.to_pickle(ecg_pickle_path)
        print(f"ECG DataFrame saved to {ecg_pickle_path}")
        print("\nProcessed PCG DataFrame:")
        print(pcg_df.head())
        pcg_df.to_pickle(pcg_pickle_path)
        print(f"PCG DataFrame saved to {pcg_pickle_path}")
    except Exception as error:
        print("An error occurred during processing:", error)