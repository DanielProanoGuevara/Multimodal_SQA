import os
import glob
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder
import preprocessing_lib as pplib
import feature_extraction_lib as ftelib
import file_process_lib as importlib

# Process all database

# Path to directories
# wav_dir = r"..\Physionet_2016_training"
# mat_dir = r"..\Physionet_2016_labels"

# Just A-folder
wav_dir = r"..\Physionet_2016_training\training-a"
mat_dir = r"..\Physionet_2016_labels\training-a-Aut"


# Collect and pair .wav and .mat Files
# Get all .wav files
wav_files = glob.glob(os.path.join(wav_dir, '**', '*.wav'), recursive=True)

# Get all .mat files
mat_files = glob.glob(os.path.join(mat_dir, '**', '*.mat'), recursive=True)

# Create Dictionaries to Map Patient ID's to File Paths

wav_dict = {}
for wav_path in wav_files:
    # Extract patient ID (e.g., 'a0001')
    patient_id = os.path.splitext(os.path.basename(wav_path))[0]
    wav_dict[patient_id] = wav_path

mat_dict = {}
for mat_path in mat_files:
    # Extract patient ID by removing suffix after '_State' and extension
    basename = os.path.basename(mat_path)
    patient_id = basename.split('_State')[0]
    mat_dict[patient_id] = mat_path

# Pair the files
# Find common patient IDs
common_patient_ids = set(wav_dict.keys()).intersection(set(mat_dict.keys()))

# Create a list of (patient_id, wav_path, mat_path) tuples
paired_files = [(patient_id, wav_dict[patient_id], mat_dict[patient_id])
                for patient_id in common_patient_ids]


# Randomize the paired files
# random.shuffle(paired_files)
# Sort the paired files by patient ID
paired_files.sort(key=lambda x: x[0])

# Split the paired files into train, test, and validation sets
train_split = int(0.8 * len(paired_files))
# test_split = int(0.8 * len(paired_files))

train_files = paired_files[:train_split]
# test_files = paired_files[train_split:test_split]
validation_files = paired_files[train_split:]


# Function to process files and create a DataFrame
def process_files(paired_files):

    # Define the minimal duration in seconds
    MIN_DURATION = 2.0

    # Initialize lists to store data
    patient_ids = []
    features_list = []
    labels_list = []

    # Process all files
    for idx, (patient_id, wav_path, mat_path) in enumerate(paired_files):
        try:
            samplerate, original_data, propagated_labels = importlib.import_physionet_2016(
                wav_path, mat_path)
            time = original_data.size / samplerate
            if time < MIN_DURATION:
                print(
                    f"Skipping Patient ID {patient_id}: Audio duration {time:.2f}s is less than the minimum required {MIN_DURATION}s.")
                continue  # Skip the file

            # Process
            data = np.copy(original_data)
            z_norm = pplib.z_score_standardization(data)

            # Resample 1kHz
            resample = pplib.downsample(z_norm, samplerate, 1000)

            # Schmidt despiking
            despiked_signal = pplib.schmidt_spike_removal(resample, 1000)

            # wavelet denoising
            wavelet_denoised = pplib.wavelet_denoise(
                despiked_signal, 5, wavelet_family='coif4', risk_estimator=pplib.val_SURE_threshold, shutdown_bands=[])

            # Butterworth bandpass filtering
            filtered_pcg = pplib.butterworth_filter(
                wavelet_denoised, 'bandpass', 2, 1000, [25, 400])

            # Feature Extraction
            # Homomorphic Envelope
            homomorphic = ftelib.homomorphic_envelope(
                filtered_pcg, 1000, 50)

            # CWT Scalogram Envelope
            cwt_morl = ftelib.c_wavelet_envelope(filtered_pcg, 1000, 50,
                                                 interest_frequencies=[40, 200])

            cwt_mexh = ftelib.c_wavelet_envelope(
                filtered_pcg, 1000, 50, wv_family='mexh',
                interest_frequencies=[40, 200])

            # 3rd decomposition DWT
            # dwt = ftelib.d_wavelet_envelope(wavelet_denoised, 1000, 50)

            # Hilbert Envelope
            hilbert_env = ftelib.hilbert_envelope(filtered_pcg, 1000, 50)

            # Label Processing
            desired_order = ['S1', 'systole', 'S2', 'diastole']
            # Extract the unique labels and reshape the labels for one-hot encoding
            unique_labels = np.unique(propagated_labels)
            # Ensure that the desired order matches the unique labels
            assert set(desired_order) == set(
                unique_labels), "The desired order does not match the unique labels"

            # Reshape the labels to a 2D array to fit the OneHotEncoder input
            propagated_labels_reshaped = propagated_labels.reshape(-1, 1)

            # Initialize the OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False,
                                    categories=[desired_order])

            # Fit and transform the labels to one-hot encoding
            # one_hot_encoded = np.abs(pplib.downsample(
            #     encoder.fit_transform(propagated_labels_reshaped), samplerate, 50))

            one_hot_encoded = encoder.fit_transform(propagated_labels_reshaped)
            one_hot_encoded = one_hot_encoded[::40, :]

            # Organize
            features = np.column_stack(
                (homomorphic, cwt_morl, cwt_mexh, hilbert_env))
            labels = one_hot_encoded

            # Append data to lists
            patient_ids.append(patient_id)
            features_list.append(features)
            labels_list.append(labels)

        except Exception as e:
            print(f"Error proccessing Patient ID {patient_id}: {e}")
            continue  # Skip the file

    # Create Dataframe
    df = pd.DataFrame({
        'Patient ID': patient_ids,
        'Features': features_list,
        'Labels': labels_list
    })
    return df


# Process and save train, test, and validate datasets
train_df = process_files(train_files)
# test_df = process_files(test_files)
validation_df = process_files(validation_files)

# Modify the DataFrame to store each feature separately
train_df['Homomorphic'] = train_df['Features'].apply(lambda x: x[:, 0])
train_df['CWT_Morl'] = train_df['Features'].apply(lambda x: x[:, 1])
train_df['CWT_Mexh'] = train_df['Features'].apply(lambda x: x[:, 2])
train_df['Hilbert_Env'] = train_df['Features'].apply(lambda x: x[:, 3])
train_df = train_df.drop(columns=['Features'])

# test_df['Homomorphic'] = test_df['Features'].apply(lambda x: x[:, 0])
# test_df['CWT_Morl'] = test_df['Features'].apply(lambda x: x[:, 1])
# test_df['CWT_Mexh'] = test_df['Features'].apply(lambda x: x[:, 2])
# test_df['Hilbert_Env'] = test_df['Features'].apply(lambda x: x[:, 3])
# test_df = test_df.drop(columns=['Features'])

validation_df['Homomorphic'] = validation_df['Features'].apply(
    lambda x: x[:, 0])
validation_df['CWT_Morl'] = validation_df['Features'].apply(lambda x: x[:, 1])
validation_df['CWT_Mexh'] = validation_df['Features'].apply(lambda x: x[:, 2])
validation_df['Hilbert_Env'] = validation_df['Features'].apply(
    lambda x: x[:, 3])
validation_df = validation_df.drop(columns=['Features'])

# Save the DataFrames to pickle files
train_pickle_path = r'..\train_physionet_2016.pkl'
# test_pickle_path = r'..\test_physionet_2016.pkl'
validation_pickle_path = r'..\validation_physionet_2016.pkl'

train_df.to_pickle(train_pickle_path)
print(f"Train DataFrame saved to {train_pickle_path}")

# test_df.to_pickle(test_pickle_path)
# print(f"Test DataFrame saved to {test_pickle_path}")

validation_df.to_pickle(validation_pickle_path)
print(f"Validate DataFrame saved to {validation_pickle_path}")
