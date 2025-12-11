# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 10:10:35 2025

Train an RBF-kernel SVM on the reduced 3-label set using the four
alignment_metric features as predictors.

This script reuses the same data loading and quantization logic used in
hypothesis_tests.py, but replaces the classifier with an SVM.

Output:
    - Console metrics (AUC, accuracy, sensitivity, specificity, F1)
    - Confusion matrix image (%)
    - Per-sample CSV report with ID, auscultation focus, manual score,
      min_lin, and decoded SVM prediction.
    - (Model artifact saving code left commented for now.)

Author: Daniel Proaño-Guevara (adapted)
"""

# %% Imports
import copy
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)

# Paths as in hypothesis_tests.py – adapt if needed
AQ_PATH = r"..\ulsge_quality_metrics.pkl"
MQ_PATH = r"..\ulsge_manual_sqa.xlsx"
MODEL_PATH = "svm_min_lin_rbf.pkl"
REPORT_PATH = "svm_min_lin_report.csv"

# %% Merge function (reused from hypothesis_tests.py)
def merge_quality_dataframes(ex1_quality: pd.DataFrame, m_quality: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ex1_quality and m_quality based on:
      1. ex1_quality['ID'] (as string) == m_quality['Trial'] (as string)
      2. ex1_quality['Auscultation_Point'] matches m_quality['Spot']
         ignoring underscores and case.

    Returns a DataFrame with:
      - 'ID'
      - 'Auscultation_Point'
      - 'mSQA_min'
      - All columns from ex1_quality that start with 'alignment_metric'
    """
    ex1 = ex1_quality.copy()
    m = m_quality.copy()

    ex1["ID"] = ex1["ID"].astype(str)
    m["Trial"] = m["Trial"].astype(str)

    ex1["Normalized_Point"] = (
        ex1["Auscultation_Point"].str.replace("_", "", regex=False).str.upper()
    )
    m["Normalized_Spot"] = m["Spot"].str.replace("_", "", regex=False).str.upper()

    merged_df = pd.merge(
        ex1,
        m[["Trial", "mSQA_min", "Normalized_Spot"]],
        left_on=["ID", "Normalized_Point"],
        right_on=["Trial", "Normalized_Spot"],
        how="inner",
    )

    alignment_metric_cols = [
        col for col in ex1.columns if col.startswith("alignment_metric")
    ]

    result_df = merged_df[["ID", "Auscultation_Point", "mSQA_min"] + alignment_metric_cols]

    return result_df


# %% 1. Load data
print("Loading quality metrics and manual annotations...")
ex1_quality_original = pd.read_pickle(AQ_PATH)
m_quality_original = pd.read_excel(MQ_PATH)

# Deep copies (optional, consistent with hypothesis_tests.py)
ex1_quality = copy.deepcopy(ex1_quality_original)
m_quality = copy.deepcopy(m_quality_original)

# %% 2. Merge into a single DataFrame
print("Merging automatic and manual quality DataFrames...")
merged_df = merge_quality_dataframes(ex1_quality, m_quality)

# %% 3. Build test_df (keep ID and Auscultation_Point for reporting)
alignment_metrics = [
    "alignment_metric_min_lin",
    "alignment_metric_avg_lin",
    "alignment_metric_min_min",
    "alignment_metric_avg_min",
]

required_cols = ["mSQA_min"] + alignment_metrics
for col in required_cols:
    if col not in merged_df.columns:
        raise ValueError(f"Required column '{col}' is missing in merged_df.")

# Include ID and Auscultation_Point so we can build the final report
base_cols = ["ID", "Auscultation_Point"]
test_df = merged_df[base_cols + required_cols].dropna()
if test_df.empty:
    raise ValueError("No valid rows available after dropping NaNs in test_df.")

print(f"Total valid samples: {len(test_df)}")

# %% 4. Quantize labels into 3 classes (reduced set)
mapping = {
    0: "low_quality",
    1: "uncertain",
    2: "uncertain",
    3: "high_quality",
    4: "high_quality",
    5: "high_quality",
}

test_df["quantized"] = test_df["mSQA_min"].map(mapping)

# Drop any rows that did not map correctly (should be none if scores are 0–5)
test_df = test_df.dropna(subset=["quantized"])
quantized_labels = ["low_quality", "uncertain", "high_quality"]

print("Quantized label distribution:")
print(test_df["quantized"].value_counts())

# %% 5. Prepare X (all 4 alignment metrics) and y (3-class)
# Features: all four alignment metrics
X = test_df[alignment_metrics].values  # shape (N, 4)
y_str = test_df["quantized"].values    # string labels

# Encode string labels -> integers
le = LabelEncoder()
y = le.fit_transform(y_str)
unique_int_labels = np.unique(y)

# Desired label order (strings)
ordered_labels_str = ["low_quality", "uncertain", "high_quality"]

# Convert to encoded integer labels in that specific order
ordered_labels_int = le.transform(ordered_labels_str)

print("\nLabel encoding mapping:")
for class_idx, class_name in enumerate(le.classes_):
    print(f"  {class_name} -> {class_idx}")

# %% 6. Scale features (recommended for RBF SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% 7. Define and train SVM (RBF kernel)
print("\nTraining RBF-kernel SVM on 4 alignment_metric features...")
svm_clf = SVC(
    kernel="rbf",
    probability=True,      # needed for ROC-AUC
    class_weight="balanced",
    random_state=42,
)

svm_clf.fit(X_scaled, y)

# %% 8. Predictions and probabilities
y_pred = svm_clf.predict(X_scaled)
y_proba = svm_clf.predict_proba(X_scaled)

# Decode predictions back to string labels for reporting
y_pred_str = le.inverse_transform(y_pred)

# Build final per-sample report
report_df = test_df.copy()
report_df["svm_pred_label"] = y_pred_str

# Keep only the requested columns in the final report
report_df = report_df[
    [
        "ID",
        "Auscultation_Point",
        "mSQA_min",
        "alignment_metric_min_lin",
        "svm_pred_label",
    ]
]

report_df.to_csv(REPORT_PATH, index=False)
print(f"\nPer-sample report saved to: {REPORT_PATH}")

# %% 9. Metrics
print("\n-------------------------------------------------------")
print("Performance metrics (evaluated on the training set)")
print("-------------------------------------------------------")

# AUC (OvR, multi-class)
y_bin = label_binarize(y, classes=unique_int_labels)
auc_ovr = roc_auc_score(y_bin, y_proba, multi_class="ovr")
print(f"AUC (OvR, multi-class): {auc_ovr:.4f}")

# Accuracy
acc = accuracy_score(y, y_pred)
print(f"Accuracy:               {acc:.4f}")

# Sensitivity / Recall (macro)
sens_macro = recall_score(y, y_pred, average="macro")
print(f"Macro Sensitivity:      {sens_macro:.4f}")

# F1-score (macro)
f1_macro = f1_score(y, y_pred, average="macro")
print(f"Macro F1-score:         {f1_macro:.4f}")

# Confusion matrix with fixed order: Low Quality, Uncertain, High quality
cm = confusion_matrix(y, y_pred, labels=ordered_labels_int)

# Per-class metrics (using cm in the fixed order)
n_classes = cm.shape[0]
tp = np.diag(cm)
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp
tn = cm.sum() - (tp + fp + fn)

# Avoid division by zero
precision_per_class = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
recall_per_class = np.where(tp + fn > 0, tp / (tp + fn), 0.0)  # sensitivity
specificity_per_class = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
f1_per_class = np.where(
    precision_per_class + recall_per_class > 0,
    2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class),
    0.0,
)
support_per_class = cm.sum(axis=1)

# Macro specificity from the per-class specificities
spec_macro = specificity_per_class.mean()
print(f"Macro Specificity:      {spec_macro:.4f}")

print("\nConfusion matrix (rows=true, cols=pred):")
print("Order: Low Quality, Uncertain, High quality")
print(cm)

# Pretty print per-class metrics in the requested order
print("\nPer-class metrics:")
print("Class              Support  Prec.   Sens.   Spec.   F1")
for idx, label_name in enumerate(ordered_labels_str):
    print(
        f"{label_name:<18} "
        f"{support_per_class[idx]:>7d}  "
        f"{precision_per_class[idx]:.4f}  "
        f"{recall_per_class[idx]:.4f}  "
        f"{specificity_per_class[idx]:.4f}  "
        f"{f1_per_class[idx]:.4f}"
    )

# Plot confusion matrix as percentages
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized,
                              display_labels=ordered_labels_str)

fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, values_format=".1f", cmap="Blues", colorbar=True)

plt.title("SVM (RBF) – Confusion Matrix (4 alignment features, %)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")

plt.tight_layout()
# plt.savefig("svm_min_lin_confusion_matrix_percent.png",
#             dpi=300, bbox_inches="tight")
# plt.close(fig)

print("\nPercentage confusion matrix figure saved as: svm_min_lin_confusion_matrix_percent.png")

# %% 10. Save model artifact (optional: uncomment if you want to persist the model)
model_artifact = {
    "svm_model": svm_clf,
    "scaler": scaler,
    "label_encoder": le,
    "quantized_labels": quantized_labels,
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_artifact, f)

print(f"\nTrained SVM model saved to: {MODEL_PATH}")
