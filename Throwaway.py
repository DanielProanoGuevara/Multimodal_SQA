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
from matplotlib.gridspec import GridSpec
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
from sklearn.preprocessing import OneHotEncoder

# import librosa
# import logging
import scipy.io
# import scipy.signal
# import re

import pickle

from scipy.io import wavfile

from scipy.stats import pearsonr, kendalltau, spearmanr


import copy

# %% Import Automatic Quality Metrics
aq_dir = r'..\ulsge_quality_metrics.pkl'
ex1_quality_original = pd.read_pickle(aq_dir)
# Deep copy
ex1_quality = copy.deepcopy(ex1_quality_original)
# %% Import Manual Annotations
mq_dir = r'..\ulsge_manual_sqa.xlsx'
m_quality_original = pd.read_excel(mq_dir)
# Deep Copy
m_quality = copy.deepcopy(m_quality_original)
# %% Merge dataframes with manually annotated metrics


def merge_quality_dataframes(ex1_quality: pd.DataFrame, m_quality: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two DataFrames, ex1_quality and m_quality, based on:
      1. ex1_quality['ID'] (as a string) equals m_quality['Trial'] (as a string)
      2. ex1_quality['Auscultation_Point'] matches m_quality['Spot'] ignoring underscores.

    The merged DataFrame will include:
      - 'ID'
      - 'Auscultation_Point'
      - 'mSQA_min' (from m_quality)
      - All columns from ex1_quality that start with 'alignment_metric'

    Only rows meeting both matching conditions are kept.

    Parameters:
        ex1_quality (pd.DataFrame): DataFrame with quality and alignment metrics.
        m_quality (pd.DataFrame): DataFrame with additional quality metrics including 'mSQA_min'.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    # Create copies to avoid modifying the original DataFrames
    ex1 = ex1_quality.copy()
    m = m_quality.copy()

    # Ensure the merge keys are of the same type by converting them to string
    ex1['ID'] = ex1['ID'].astype(str)
    m['Trial'] = m['Trial'].astype(str)

    # Normalize auscultation point strings for robust matching:
    # Remove underscores and convert to uppercase for both DataFrames.
    ex1['Normalized_Point'] = ex1['Auscultation_Point'].str.replace(
        '_', '', regex=False).str.upper()
    m['Normalized_Spot'] = m['Spot'].str.replace(
        '_', '', regex=False).str.upper()

    # Merge DataFrames based on 'ID' and normalized auscultation point columns.
    merged_df = pd.merge(
        ex1,
        m[['Trial', 'mSQA_min', 'Normalized_Spot']],
        left_on=['ID', 'Normalized_Point'],
        right_on=['Trial', 'Normalized_Spot'],
        how='inner'
    )

    # Identify all columns from ex1_quality that start with 'alignment_metric'
    alignment_metric_cols = [
        col for col in ex1.columns if col.startswith('alignment_metric')]

    # Build the resulting DataFrame with the required columns:
    result_df = merged_df[['ID', 'Auscultation_Point',
                           'mSQA_min'] + alignment_metric_cols]

    return result_df


merged_df = merge_quality_dataframes(ex1_quality, m_quality)
merged_df = merged_df.dropna(subset=['mSQA_min'])


def visualize_and_analyze(merged_df: pd.DataFrame) -> None:
    """
    Process the merged_df to produce violin plots and compute correlations.

    Visualization:
      - For each alignment metric column:
         - Generate a figure containing violin plots for each mSQA_min category (0,1,2,3,4,5).
         - The x-axis represents the mSQA_min categories and the y-axis represents the metric values.
         - Axes and titles are clearly labeled.

    Statistical Analysis:
      - Compute Pearson, Kendall Tau, and Spearman correlations between mSQA_min and each alignment metric.
      - Print the correlation coefficients and p-values in a clear format.

    Parameters:
        merged_df (pd.DataFrame): DataFrame containing at least the following columns:
            - 'mSQA_min'
            - One or more alignment metric columns: 'alignment_metric_min_lin', 'alignment_metric_avg_lin', 
              'alignment_metric_min_min', 'alignment_metric_avg_min'
    """
    # List of alignment metric columns to process
    alignment_metrics = [
        'alignment_metric_min_lin',
        'alignment_metric_avg_lin',
        'alignment_metric_min_min',
        'alignment_metric_avg_min'
    ]

    # Define the mSQA_min categories (assumed to be 0 through 5)
    mSQA_min_categories = [0, 1, 2, 3, 4, 5]

    # Ensure required columns exist; if not, raise an error
    required_cols = ['mSQA_min'] + alignment_metrics
    for col in required_cols:
        if col not in merged_df.columns:
            raise ValueError(
                f"Column '{col}' is missing from the merged DataFrame.")

    # Remove rows with NaN values in the necessary columns for plotting and correlation analysis
    plot_df = merged_df.dropna(subset=required_cols)
    if plot_df.empty:
        print("No valid data available after removing NaN values. Exiting.")
        return

    # Visualization: Violin Plots
    for metric in alignment_metrics:
        # Create a new figure for the current alignment metric
        fig, ax = plt.subplots(figsize=(10, 6))

        # Prepare data: list of arrays corresponding to each mSQA_min category
        data = []
        for category in mSQA_min_categories:
            # Select rows where mSQA_min equals the current category, then extract the metric values
            cat_data = plot_df[plot_df['mSQA_min']
                               == category][metric].dropna().values
            data.append(cat_data)

        # Create the violin plot:
        # The positions are set to 1-indexed positions for clarity on the x-axis.
        parts = ax.violinplot(data, showmeans=True, showmedians=True)

        # Set x-tick labels to match the mSQA_min categories
        ax.set_xticks(range(1, len(mSQA_min_categories) + 1))
        ax.set_xticklabels(mSQA_min_categories)

        # Set axis labels and title
        ax.set_xlabel("mSQA_min Category")
        ax.set_ylabel(metric)
        ax.set_title(f"Violin Plot for {metric}")
        ax.grid(True)

        # Display the figure
        plt.show()

    # Statistical Analysis: Correlations
    print("Statistical Analysis: Correlations between mSQA_min and alignment metrics")
    for metric in alignment_metrics:
        # Drop NaN values for the two columns used in correlation computation
        corr_df = merged_df[['mSQA_min', metric]].dropna()
        if corr_df.empty:
            print(f"\nNo valid data to compute correlations for {metric}.")
            continue

        # Extract series for correlation
        mSQA_vals = corr_df['mSQA_min']
        metric_vals = corr_df[metric]

        # Compute correlations with error handling
        try:
            pearson_corr, p_pearson = pearsonr(mSQA_vals, metric_vals)
        except Exception as e:
            pearson_corr, p_pearson = (None, None)
            print(f"Error computing Pearson correlation for {metric}: {e}")

        try:
            kendall_corr, p_kendall = kendalltau(mSQA_vals, metric_vals)
        except Exception as e:
            kendall_corr, p_kendall = (None, None)
            print(f"Error computing Kendall tau correlation for {metric}: {e}")

        try:
            spearman_corr, p_spearman = spearmanr(mSQA_vals, metric_vals)
        except Exception as e:
            spearman_corr, p_spearman = (None, None)
            print(f"Error computing Spearman correlation for {metric}: {e}")

        # Print results for the current metric
        print(f"\nCorrelation results for {metric}:")
        if pearson_corr is not None:
            print(
                f"  Pearson correlation: {pearson_corr:.3f} (p-value: {p_pearson:.3g})")
        else:
            print("  Pearson correlation: Error in computation.")
        if kendall_corr is not None:
            print(
                f"  Kendall tau correlation: {kendall_corr:.3f} (p-value: {p_kendall:.3g})")
        else:
            print("  Kendall tau correlation: Error in computation.")
        if spearman_corr is not None:
            print(
                f"  Spearman correlation: {spearman_corr:.3f} (p-value: {p_spearman:.3g})")
        else:
            print("  Spearman correlation: Error in computation.")


visualize_and_analyze(merged_df)
