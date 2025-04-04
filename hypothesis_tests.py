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
from scipy import stats

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize, LabelEncoder

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

# %%
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

# # Visualization: Violin Plots
# for metric in alignment_metrics:
#     # Create a new figure for the current alignment metric
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Prepare data: list of arrays corresponding to each mSQA_min category
#     data = []
#     for category in mSQA_min_categories:
#         # Select rows where mSQA_min equals the current category, then extract the metric values
#         cat_data = plot_df[plot_df['mSQA_min']
#                            == category][metric].dropna().values
#         data.append(cat_data)

#     # Create the violin plot:
#     # The positions are set to 1-indexed positions for clarity on the x-axis.
#     parts = ax.violinplot(data, showmeans=True, showmedians=True)

#     # Set x-tick labels to match the mSQA_min categories
#     ax.set_xticks(range(1, len(mSQA_min_categories) + 1))
#     ax.set_xticklabels(mSQA_min_categories)

#     # Set axis labels and title
#     ax.set_xlabel("mSQA_min Category")
#     ax.set_ylabel(metric)
#     ax.set_title(f"Violin Plot for {metric}")
#     ax.grid(True)

#     # Display the figure
#     plt.show()

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

# %%
print("\n------------------------------------------------------------------\n")
print("Hypothesis tests conducted exclussively on the min_lin metric.\n")
# Hypothesis testing -- Best reults min-lin lambda = 0.1 unsequenced
# Create Final testing df
test_df = merged_df[['mSQA_min', 'alignment_metric_min_lin', 'alignment_metric_avg_lin', 'alignment_metric_min_min', 'alignment_metric_avg_min']].dropna()

###############################################################################
# Calculate Mean and Variance for Original Groups
###############################################################################
# Group by 'score' and compute mean and variance.
original_stats = test_df.groupby('mSQA_min')['alignment_metric_min_lin'].agg(['mean', 'var']).reset_index()
print("Original Groups - Mean and Variance:")
print(original_stats)
print()
print("Kruskal–Wallis Test selected due to the lack of uniformity in the variances, and the impossibility to assume a normal distribution of the data.\n")

# %%
###############################################################################
# Task 1: Kruskal–Wallis Test (Original Groups)
###############################################################################
# Group the data by the 'score' column.
# Note: The Kruskal–Wallis test is a non-parametric method which does not assume normality 
# and is robust to unbalanced group sizes.
unique_groups = sorted(test_df['mSQA_min'].unique())
group_data_list = [test_df[test_df['mSQA_min'] == grp]['alignment_metric_min_lin'] for grp in unique_groups]

# Perform the Kruskal–Wallis test for independent samples.
H_stat, p_val = stats.kruskal(*group_data_list)

# Print the Kruskal–Wallis test results in scientific notation.
print("Task 1: Kruskal–Wallis Test (Original Groups)")
print(f"H-statistic: {H_stat:.4e}")
print(f"p-value: {p_val:.4e}")
print("\n# Note: The Kruskal–Wallis test is non-parametric and suitable for unbalanced groups.")

# %%
###############################################################################
# Task 2: Pairwise T-tests with Bonferroni Correction (Original Groups)
###############################################################################
print("\nTask 2: Pairwise T-tests with Bonferroni Correction (Original Groups)")

# Initialize a list to hold the results and a counter for the number of comparisons.
pairwise_results = []
num_comparisons = 0

# Loop through all unique pairs of groups for independent two-sample t-tests.
# Using equal_var=False to handle possible variance heterogeneity due to unbalanced sample sizes.
for i in range(len(unique_groups)):
    for j in range(i + 1, len(unique_groups)):
        grp1 = unique_groups[i]
        grp2 = unique_groups[j]
        data1 = test_df[test_df['mSQA_min'] == grp1]['alignment_metric_min_lin']
        data2 = test_df[test_df['mSQA_min'] == grp2]['alignment_metric_min_lin']
        t_stat, p_val_pair = stats.ttest_ind(data1, data2, equal_var=False)
        pairwise_results.append((f"{grp1} vs {grp2}", p_val_pair))
        num_comparisons += 1

# Print the total number of pairwise comparisons.
print("Total pairwise comparisons:", num_comparisons)

# Apply Bonferroni correction and print results for each pair.
for comparison, orig_p in pairwise_results:
    # Multiply the original p-value by the number of comparisons; cap the value at 1.
    corrected_p = min(orig_p * num_comparisons, 1.0)
    significance = "Significant" if corrected_p < 0.05 else "Not significant"
    print(f"{comparison}: Original p-value = {orig_p:.4e}, Bonferroni-corrected p-value = {corrected_p:.4e} -> {significance}")

# %%
###############################################################################
# Task 3: Group Quantization and New Kruskal–Wallis Test
###############################################################################
# Create a new column 'quantized' to re-label groups into three categories:
#   - Groups 0 and 1 -> "Low_quality"
#   - Groups 2 and 3 -> "uncertain"
#   - Groups 4 and 5 -> "high_quality"
# Using np.select to avoid function definitions.
conditions = [
    test_df['mSQA_min'].isin([0, 1]),
    test_df['mSQA_min'].isin([2, 3]),
    test_df['mSQA_min'].isin([4, 5])
]
choices = ["Low_quality", "uncertain", "high_quality"]
test_df['quantized'] = np.select(conditions, choices, default=np.nan)

# Calculate Mean and Variance for Quantized Groups
quantized_stats = test_df.groupby('quantized')['alignment_metric_min_lin'].agg(['mean', 'var']).reset_index()
print("\nQuantized Groups - Mean and Variance:")
print(quantized_stats)
print()

# Group the data by the new 'quantized' column.
unique_quant = sorted(test_df['quantized'].unique())
quant_group_data_list = [test_df[test_df['quantized'] == grp]['alignment_metric_min_lin'] for grp in unique_quant]

# Perform the Kruskal–Wallis test on the quantized groups.
H_stat_quant, p_val_quant = stats.kruskal(*quant_group_data_list)

# Print the Kruskal–Wallis test results for quantized groups in scientific notation.
print("Task 3: Kruskal–Wallis Test (Quantized Groups)")
print(f"H-statistic: {H_stat_quant:.4e}")
print(f"p-value: {p_val_quant:.4e}")
print("\n# Note: The Kruskal–Wallis test is non-parametric and suitable for unbalanced groups.")

# %%
###############################################################################
# Task 4: Pairwise T-tests with Bonferroni Correction (Quantized Groups)
###############################################################################
print("\nTask 4: Pairwise T-tests with Bonferroni Correction (Quantized Groups)")

# Initialize a list to hold quantized pairwise t-test results.
quant_pairwise_results = []
num_quant_comparisons = 0

# Loop through all unique pairs of quantized groups.
for i in range(len(unique_quant)):
    for j in range(i + 1, len(unique_quant)):
        grp1 = unique_quant[i]
        grp2 = unique_quant[j]
        data1 = test_df[test_df['quantized'] == grp1]['alignment_metric_min_lin']
        data2 = test_df[test_df['quantized'] == grp2]['alignment_metric_min_lin']
        t_stat, p_val_pair = stats.ttest_ind(data1, data2, equal_var=False)
        quant_pairwise_results.append((f"{grp1} vs {grp2}", p_val_pair))
        num_quant_comparisons += 1

# Print the total number of quantized pairwise comparisons.
print("Total quantized pairwise comparisons:", num_quant_comparisons)

# Apply Bonferroni correction and print the results.
for comparison, orig_p in quant_pairwise_results:
    corrected_p = min(orig_p * num_quant_comparisons, 1.0)
    significance = "Significant" if corrected_p < 0.05 else "Not significant"
    print(f"{comparison}: Original p-value = {orig_p:.4e}, Bonferroni-corrected p-value = {corrected_p:.4e} -> {significance}")

# %%

# =============================================================================
# Task 5: Multinomial Logistic Regression & AUC Evaluation
# =============================================================================
# We perform multinomial logistic regression for two settings:
# 1. Using the original (unquantized) dataset: outcome = 'score' (0 to 5)
# 2. Using the quantized dataset: outcome = 'quantized'
# For each setting, we:
#    - Train on the whole dataset and evaluate AUC on the same data.
#    - Perform a 20-80 train-test split and evaluate the test AUC.
#
# Note: For multiclass AUC, we use roc_auc_score with multi_class='ovr'.
#
# ------------------------------
# 1. Unquantized Dataset
# ------------------------------

# Define the four features to be used in the multinomial regression.
features = ['alignment_metric_min_lin', 'alignment_metric_avg_lin', 
            'alignment_metric_min_min', 'alignment_metric_avg_min']


print("\nTask 5: Multinomial Logistic Regression & AUC Evaluation on Unquantized Data")

# Prepare features and outcome.
X_unq = test_df[features].values
y_unq = test_df['mSQA_min'].values  # Outcome: integers 0-5

# a) Train on the whole dataset.
model_unq = LogisticRegressionCV(cv=5, class_weight='balanced', max_iter=1000)
model_unq.fit(X_unq, y_unq)
y_pred_proba_unq = model_unq.predict_proba(X_unq)
# Binarize true labels for AUC computation.
y_unq_bin = label_binarize(y_unq, classes=np.unique(y_unq))
auc_unq = roc_auc_score(y_unq_bin, y_pred_proba_unq, multi_class='ovr')
print("Whole Dataset:")
print(f"  AUC: {auc_unq:.4e}")

# Compute and print the confusion matrix for the unquantized dataset.
y_pred_unq = model_unq.predict(X_unq)
cm_unq = confusion_matrix(y_unq, y_pred_unq)
print("Confusion Matrix for Unquantized Data:")
print(cm_unq)

# ------------------------------
# 2. Quantized Dataset
# ------------------------------
print("\nTask 5: Multinomial Logistic Regression & AUC Evaluation on Quantized Data")

# Prepare features and outcome.
# Outcome: 'quantized' (categories: "Low_quality", "uncertain", "high_quality")
# We need to convert string labels to numeric labels.
le = LabelEncoder()
y_quant = le.fit_transform(test_df['quantized'])
X_quant = test_df[features].values

# a) Train on the whole dataset.
model_quant = LogisticRegressionCV(cv=5, class_weight='balanced', max_iter=1000)
model_quant.fit(X_quant, y_quant)
y_pred_proba_quant = model_quant.predict_proba(X_quant)
y_quant_bin = label_binarize(y_quant, classes=np.unique(y_quant))
auc_quant = roc_auc_score(y_quant_bin, y_pred_proba_quant, multi_class='ovr')
print("Whole Dataset:")
print(f"  AUC: {auc_quant:.4e}")

# Compute and print the confusion matrix for the quantized dataset.
y_pred_quant = model_quant.predict(X_quant)
cm_quant = confusion_matrix(y_quant, y_pred_quant)
print("Confusion Matrix for Quantized Data:")
print(cm_quant)


# %%
