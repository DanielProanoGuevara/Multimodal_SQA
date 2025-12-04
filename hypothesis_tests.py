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

import seaborn as sns

# import librosa
# import logging
import scipy.io
from scipy import stats

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (roc_auc_score, 
                             confusion_matrix, 
                             ConfusionMatrixDisplay, 
                             accuracy_score,
                             f1_score,
                             recall_score)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize, LabelEncoder


from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import softmax
from scipy.stats import norm


from sklearn.inspection import permutation_importance

# import scipy.signal
# import re

import pickle

from scipy.io import wavfile

from scipy.stats import pearsonr, kendalltau, spearmanr, norm


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

# Visualization: Violin Plots
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

fig, ax = plt.subplots(figsize=(4, 4), )
data = []
for category in mSQA_min_categories:
    # Select rows where mSQA_min equals the current category, then extract the metric values
    cat_data = plot_df[plot_df['mSQA_min']
                       == category]['alignment_metric_min_lin'].dropna().values
    data.append(cat_data)
parts = ax.violinplot(data, showmeans=True, showmedians=True)
ax.set_xticks(range(1, len(mSQA_min_categories) + 1))
ax.set_xticklabels(mSQA_min_categories)
ax.set_xlabel("mSQA_min Category")
ax.set_ylabel('SQI min lin')
ax.set_title("SQI with $\lambda = 0.1$")
ax.grid(True)
plt.tight_layout()
plt.show()
# plt.savefig('SQI lambda 01.pdf', format='pdf', dpi=900)


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
test_df = merged_df[['mSQA_min', 'alignment_metric_min_lin', 'alignment_metric_avg_lin',
                     'alignment_metric_min_min', 'alignment_metric_avg_min']].dropna()

###############################################################################
# Calculate Mean and Variance for Original Groups
###############################################################################
# Group by 'score' and compute mean and variance.
original_stats = test_df.groupby('mSQA_min')['alignment_metric_min_lin'].agg([
    'mean', 'var']).reset_index()
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
group_data_list = [test_df[test_df['mSQA_min'] == grp]
                   ['alignment_metric_min_lin'] for grp in unique_groups]

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
        data1 = test_df[test_df['mSQA_min'] ==
                        grp1]['alignment_metric_min_lin']
        data2 = test_df[test_df['mSQA_min'] ==
                        grp2]['alignment_metric_min_lin']
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
mapping = {
    0: "low_quality",
    1: "uncertain",
    2: "uncertain",
    3: "high_quality",
    4: "high_quality",
    5: "high_quality",
}

test_df['quantized'] = test_df['mSQA_min'].map(mapping)

# Calculate Mean and Variance for Quantized Groups
quantized_stats = test_df.groupby('quantized')['alignment_metric_min_lin'].agg([
    'mean', 'var']).reset_index()
print("\nQuantized Groups - Mean and Variance:")
print(quantized_stats)
print()

# Group the data by the new 'quantized' column.
unique_quant = sorted(test_df['quantized'].unique())
quant_group_data_list = [test_df[test_df['quantized'] == grp]
                         ['alignment_metric_min_lin'] for grp in unique_quant]

# Perform the Kruskal–Wallis test on the quantized groups.
H_stat_quant, p_val_quant = stats.kruskal(*quant_group_data_list)

# Print the Kruskal–Wallis test results for quantized groups in scientific notation.
print("Task 3: Kruskal–Wallis Test (Quantized Groups)")
print(f"H-statistic: {H_stat_quant:.4e}")
print(f"p-value: {p_val_quant:.4e}")
print("\n# Note: The Kruskal–Wallis test is non-parametric and suitable for unbalanced groups.")

fig, ax = plt.subplots(figsize=(4, 4))
quantized_labels = ["low_quality", "uncertain", "high_quality"]

# Extract data for each quantized group
quantized_data = [test_df[test_df['quantized'] == label]['alignment_metric_min_lin'].dropna().values
                  for label in quantized_labels]

# Create violin plot
parts = ax.violinplot(quantized_data, showmeans=True, showmedians=True)

# Set x-tick labels
ax.set_xticks(range(1, len(quantized_labels) + 1))
ax.set_xticklabels(quantized_labels)

# Set axis labels and title
ax.set_ylabel("SQI min lin")
ax.set_title("Relabeled SQI Distribution")
ax.grid(True)
plt.tight_layout()
plt.show()
# plt.savefig('SQI relabled.pdf', format='pdf', dpi=900)

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
        data1 = test_df[test_df['quantized'] ==
                        grp1]['alignment_metric_min_lin']
        data2 = test_df[test_df['quantized'] ==
                        grp2]['alignment_metric_min_lin']
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

# %%
###############################################################################
# Task 5: Multinomial Logistic Regression with Feature Importance
###############################################################################


def summarize_coefficients(model, feature_names, label_encoder=None):
    """
    Prints the average magnitude of each feature's coefficient across all classes.
    """
    coef = model.coef_  # shape: (n_classes, n_features)
    abs_mean = np.mean(np.abs(coef), axis=0)
    print("\n--- Feature Importance Summary (|coef| averaged across classes) ---")
    for fname, score in zip(feature_names, abs_mean):
        print(f"  {fname:30s}: {score:.4f}")
    best_idx = np.argmax(abs_mean)
    print(f"\nMost relevant feature: {feature_names[best_idx]}")


def compute_permutation_importance(model, X, y, feature_names, title="Permutation Importance"):
    """
    Computes permutation importance and prints sorted results.
    """
    print(f"\n--- {title} ---")
    result = permutation_importance(
        model, X, y, n_repeats=30, random_state=42, n_jobs=-1)

    importances = result.importances_mean
    std = result.importances_std
    indices = np.argsort(importances)[::-1]

    for idx in indices:
        print(
            f"{feature_names[idx]:30s} | Importance: {importances[idx]:.4f} ± {std[idx]:.4f}")

    best_idx = indices[0]
    print(
        f"\nMost relevant input: {feature_names[best_idx]} (Permutation-based)")


# -----------------------------
# Prepare data and features
# -----------------------------
features = ['alignment_metric_min_lin', 'alignment_metric_avg_lin',
            'alignment_metric_min_min', 'alignment_metric_avg_min']
X_unq = test_df[features].values
y_unq = test_df['mSQA_min'].values
le_quant = LabelEncoder()
y_quant = le_quant.fit_transform(test_df['quantized'])
X_quant = test_df[features].values

# -----------------------------
# Unquantized Model
# -----------------------------
print("\nTask 5a: Multinomial Logistic Regression on Unquantized Data")

model_unq = LogisticRegression(max_iter=1000, class_weight='balanced')
model_unq.fit(X_unq, y_unq)

y_pred_unq = model_unq.predict(X_unq)
y_proba_unq = model_unq.predict_proba(X_unq)
auc_unq = roc_auc_score(label_binarize(
    y_unq, classes=np.unique(y_unq)), y_proba_unq, multi_class='ovr')
print(f"  AUC (Unquantized): {auc_unq:.4f}")

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_unq, y_pred_unq),
                              display_labels=np.unique(y_unq))


disp.plot()
plt.tight_layout()
plt.show()

summarize_coefficients(model_unq, features)
compute_permutation_importance(
    model_unq, X_unq, y_unq, features, title="Unquantized Permutation Importance")


# -----------------------------
# Quantized Model
# -----------------------------
print("\nTask 5b: Multinomial Logistic Regression on Quantized Data")

model_quant = LogisticRegression(max_iter=1000, class_weight='balanced')
model_quant.fit(X_quant, y_quant)

y_pred_quant = model_quant.predict(X_quant)
y_proba_quant = model_quant.predict_proba(X_quant)

# AUC (OvR, multi-class)
auc_quant = roc_auc_score(
    label_binarize(y_quant, classes=np.unique(y_quant)),
    y_proba_quant,
    multi_class='ovr'
)
print(f"  AUC (Quantized): {auc_quant:.4f}")

# -----------------------------
# Additional metrics (3-class, macro-averaged)
# -----------------------------
# Accuracy
acc_quant = accuracy_score(y_quant, y_pred_quant)

# Sensitivity (Recall) – macro average over classes
sens_macro = recall_score(y_quant, y_pred_quant, average='macro')

# F1 – macro average over classes
f1_macro = f1_score(y_quant, y_pred_quant, average='macro')

# Specificity – computed per class from confusion matrix, then macro average
labels_int = np.unique(y_quant)
cm_int = confusion_matrix(y_quant, y_pred_quant, labels=labels_int)

# cm_int[i, j] = true class i, predicted class j
tp = np.diag(cm_int)
fp = cm_int.sum(axis=0) - tp
fn = cm_int.sum(axis=1) - tp
tn = cm_int.sum() - (tp + fp + fn)

specificity_per_class = tn / (tn + fp)
spec_macro = specificity_per_class.mean()

print(f"  Accuracy (Quantized):      {acc_quant:.4f}")
print(f"  Macro Sensitivity/Recall:  {sens_macro:.4f}")
print(f"  Macro Specificity:         {spec_macro:.4f}")
print(f"  Macro F1-score:            {f1_macro:.4f}")

# -----------------------------
# Decode to string labels for presentation
# -----------------------------
y_true_labels = le_quant.inverse_transform(y_quant)
y_pred_labels = le_quant.inverse_transform(y_pred_quant)

# Create confusion matrix with explicit label order (string labels)
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=quantized_labels)

# Normalize per true class (rows)
cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
cm_percent = np.int8(cm_percent)

# Plot
fig, ax = plt.subplots(figsize=(4, 4))
sns.heatmap(
    cm_percent,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=quantized_labels,
    yticklabels=quantized_labels,
    ax=ax
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix of min_lin")
plt.tight_layout()
plt.show()
# plt.savefig('CM_SQI_relabled.pdf', format='pdf', dpi=900)

summarize_coefficients(model_quant, features)
compute_permutation_importance(
    model_quant, X_quant, y_quant, features, title="Quantized Permutation Importance"
)


# %%
###############################################################################
# Statistical Analysis: Correlations between Quantized Labels and Alignment Metrics
###############################################################################

print("\nStatistical Analysis: Correlations between quantized labels and alignment metrics")

# Encode string labels as ordinal integers for correlation
quant_corr_df = test_df[['quantized'] + alignment_metrics].dropna()
le_corr = LabelEncoder()
quant_corr_df['quantized_encoded'] = le_corr.fit_transform(
    quant_corr_df['quantized'])

for metric in alignment_metrics:
    metric_vals = quant_corr_df[metric]
    quant_vals = quant_corr_df['quantized_encoded']

    try:
        pearson_corr, p_pearson = pearsonr(quant_vals, metric_vals)
    except Exception as e:
        pearson_corr, p_pearson = (None, None)
        print(f"Error computing Pearson correlation for {metric}: {e}")

    try:
        kendall_corr, p_kendall = kendalltau(quant_vals, metric_vals)
    except Exception as e:
        kendall_corr, p_kendall = (None, None)
        print(f"Error computing Kendall tau correlation for {metric}: {e}")

    try:
        spearman_corr, p_spearman = spearmanr(quant_vals, metric_vals)
    except Exception as e:
        spearman_corr, p_spearman = (None, None)
        print(f"Error computing Spearman correlation for {metric}: {e}")

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
