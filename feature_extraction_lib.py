"""
ECG and PCG feature extraction library.

@file: feature_extraction_lib.py

@coding: utf_8

@description: This module intends to contain all the feature extraction
functions necessary for ECG and PCG signals. Also the training patches
creation.

@functions:
- homomorphic_envelope: Extracts the homomorphic envelope based on the AM
modulation, where the envelope in the modulation signal, and the carrier is to
be descarded. The signal s(t) is composed by the modulation signal e(t) and the
carrier by c(t) by: s(t)=e(t)c(t). Its separation is acheived with the
logarithmic law of multiplications.

- c_wavelet_envelope: Extracts the continuous wavelet transformation, rendering
a scalogram in a specified frequency range, then these values are averaged. The
methodology is inspired in the PSD of Spinger, 2016.

- d_wavelet_envelope: Returns the specified level wavelet decimated
decomposition, as the envelope.

- def hilbert_envelope: Computes the Hilbert transform, obtains the magnitude
of the complex results, defining it as the envelope.

- create_patches: Create the input vectors for Supervized training algorithms
and the output validation one.

- process_dataset: Use this function to preprocess your dataset by extracting
features and labels, creating patches from them, and aggregating all patches
into single arrays for training.

- reconstruct_patches: After making predictions on patches, use this function
to reconstruct the original sequence by combining the patches and averaging
overlapping regions.

- reconstruct_original_data: When dealing with multiple sequences
(e.g.,multiple patients), this function helps reconstruct each original
sequence from the patched data.

- reverse_one_hot_encoding: Use this to convert one-hot encoded labels back to
their original integer labels, especially when a specific label order was used
during encoding.

max_temporal_modelling: Apply this function to enforce temporal consistency in
predicted sequences, ensuring that state transitions follow a defined order.


@author: Daniel Proaño-Guevara.

@creationDate: 2024-07-19

@version: 0.1
"""


import numpy as np
from scipy.signal import firwin, resample_poly
from scipy import signal
from preprocessing_lib import downsample, min_max_norm
import pywt


def homomorphic_envelope(data, fs_inicial, fs_final, epsilon=0.01,
                         median_window=51):
    """
    Compute the homomorphic envelope.

    Parameters
    ----------
    data : numpy.ndarray
        Input signal to obtain the envelope.
    fs_inicial : int
        original sampling frequency.
    fs_final : int
        final sampling frequency.
    epsilon : float
        replace zero value, corrects logarithm error.
    median_window : int (odd)
        kernel size for median filter.

    Returns
    -------
    numpy.ndarray
        homomorphic envelope of passed data.

    """
    # energy_signal = data**2
    energy_signal = np.abs(data)
    energy = np.where(energy_signal == 0, epsilon, energy_signal)
    g = np.log(energy)
    envelope_log = signal.medfilt(g, median_window)
    envelope = np.exp(envelope_log)

    return min_max_norm(downsample(envelope, fs_inicial, fs_final))
    # return (envelope)


def c_wavelet_envelope(data, fs_inicial, fs_final, wv_family='morl',
                       interest_frequencies=[40, 60]):
    """
    Compute the wavelet envelope based on CWT.

    Parameters
    ----------
    data : numpy.ndarray
        Input signal to obtain the envelope.
    fs_inicial : int
        original sampling frequency.
    fs_final : int
        final sampling frequency.
    wv_family : String, optional
        Continuous wavelet family. The default is 'morl'.
    interest_frequencies : list, optional
        Pair frequencies that represent the interest range to be analyzed,
        first the lower frequency, last the highest frequency.
        The default is [50, 200].

    Returns
    -------
    numpy.ndarray
        CWT-based envelope of passed data.

    """
    # Create an array of nomralized frequencies of interest for the continuous
    # wavelet transform
    frequencies = np.geomspace(interest_frequencies[0],
                               interest_frequencies[1], num=20)/fs_inicial
    # Create the scales from the frequencies
    scales = pywt.frequency2scale(wv_family, frequencies)
    # Extract the wavelet decomposition with CWT
    decomposition, _ = pywt.cwt(data, scales, wv_family)
    # Obtain the family of envelopes
    envelopes = np.abs(decomposition)
    # Combine the envelopes into a unique one
    envelope = np.mean(envelopes, axis=0)
    return min_max_norm(downsample(envelope, fs_inicial, fs_final))
    # return envelope


def d_wavelet_envelope(data, fs_inicial, fs_final, wv_family='rbio3.9',
                       level=3):
    """
    Compute the wavelet envelope based on DWT.

    Parameters
    ----------
    data : numpy.ndarray
        Input signal to obtain the envelope.
    fs_inicial : int
        original sampling frequency.
    fs_final : int
        final sampling frequency.
    wv_family : String, optional
        Continuous wavelet family. The default is 'rbio3.9'.
    level : int, optional
        Decomposition level for the lowest-frequency detail. The default is 3.

    Returns
    -------
    numpy.ndarray
        DWT-based envelope of passed data.

    """
    wv_detail = pywt.downcoef(part='d', data=data, wavelet=wv_family,
                              level=level, mode='zero')
    envelope = np.abs(wv_detail)
    return min_max_norm(downsample(envelope, fs_inicial/(2**level), fs_final)
                        [3:-4])


def hilbert_envelope(data, fs_inicial, fs_final):
    """
    Compute the Hilbert-transform-based envelope.

    Parameters
    ----------
    data : numpy.ndarray
        Input signal to obtain the envelope.
    fs_inicial : int
        original sampling frequency.
    fs_final : int
        final sampling frequency.

    Returns
    -------
    numpy.ndarray
        Hilbert transform-based envelope of passed data.

    """
    analytic_signal = signal.hilbert(data)
    envelope = np.abs(analytic_signal)
    return min_max_norm(downsample(envelope, fs_inicial, fs_final))


def shannon_envelopenergy(signal, fs_inicial, fs_final):
    """
    Compute the Shannon Energy Envelope of a signal.

    This function calculates the Shannon Energy Envelope of a given signal by:
    1. Computing the squared energy of the signal.
    2. Applying the Shannon entropy formula to the energy.
    3. Downsampling the resulting envelope from the initial sampling frequency (`fs_inicial`) 
       to the desired final sampling frequency (`fs_final`).
    4. Normalizing the downsampled envelope using min-max normalization.

    Parameters:
    ----------
    signal : numpy.ndarray
        The input signal to process.
    fs_inicial : int or float
        The initial sampling frequency of the signal.
    fs_final : int or float
        The desired final sampling frequency for the envelope.

    Returns:
    -------
    numpy.ndarray
        The normalized, downsampled Shannon Energy Envelope of the input signal.

    Notes:
    ------
    - A small constant (`epsilon`) is added to prevent the logarithm of zero.
    - The envelope is normalized to the range [0, 1] using min-max normalization.
    """
    epsilon = 0.0001
    energy = signal ** 2
    # energy = np.abs(signal)
    envelope = np.abs(-energy * np.log(energy + epsilon))

    return min_max_norm(downsample(envelope, fs_inicial, fs_final))


def hamming_smooth_envelope(signal, window_size, fs_inicial, fs_final):
    """
    Compute the smoothed envelope of a signal using a Hamming window.

    This function smooths the input signal by applying a double-pass convolution
    with a Hamming window. The process involves:
    1. A forward pass: Convolution of the padded signal with the Hamming window.
    2. A backward pass: Convolution of the reversed forward pass result with the same Hamming window.
    3. Normalization and downsampling of the resulting envelope.

    Parameters:
    ----------
    signal : numpy.ndarray
        The input signal to process.
    window_size : int
        The size of the Hamming window. Must be a positive odd integer. If not, it will be adjusted.
    fs_inicial : int or float
        The initial sampling frequency of the signal.
    fs_final : int or float
        The desired final sampling frequency for the envelope.

    Returns:
    -------
    numpy.ndarray
        The normalized, downsampled smoothed envelope of the input signal.

    Notes:
    ------
    - The signal is padded at both ends using edge values to minimize boundary effects during convolution.
    - A double-pass smoothing approach ensures a symmetrical response.
    - The resulting envelope is normalized to the range [0, 1] using min-max normalization.

    """
    # Ensure the window size is a positive odd integer
    if window_size % 2 == 0 or window_size < 1:
        window_size = window_size + 1

    # Define the smoothing window
    window = np.hamming(window_size)

    # Signal Energy
    x = np.abs(signal)

    # Pad the array by mirroring at both ends
    pad_size = window_size - 1
    x_padded = np.pad(x, pad_size, mode='edge')

    # Forward pass: Convolve with the moving average window
    forward_pass = np.convolve(x_padded, window, mode='valid')

    # Reverse the forward pass result
    forward_pass_reversed = forward_pass[::-1]

    # Backward pass: Convolve again with the moving average window
    backward_pass = np.convolve(forward_pass_reversed, window, mode='valid')

    # Reverse the backward pass result to restore original order
    envelope = backward_pass[::-1]

    return min_max_norm(downsample(envelope, fs_inicial, fs_final))


def create_patches(Features, Labels, Patch_Size, Stride):
    """
    Create overlapping patches from Features and Labels for ANN training.

    Parameters
    ----------
    Features : numpy.ndarray
        Input feature matrix of size (Total_Samples, Num_Features).
    Labels : numpy.ndarray
        Input labels matrix of size (Total_Samples, Num_Labels).
    Patch_Size : int
        The number of samples for each patch.
    Stride : int
        The number of samples to stride between patches.

    Raises
    ------
    ValueError
        Checks input errors.

    Returns
    -------
    Features_Patch : numpy.ndarray
        Patches of features of size (Patch_Samples, Num_Features, Num_Patches).
    Labels_Patch : numpy.ndarray
        Patches of labels of size (Patch_Samples, Num_Labels, Num_Patches).

    """
    # Features = np.array(Features).T
    total_samples = Features.shape[0]
    num_features = Features.shape[1]
    num_labels = Labels.shape[1] if len(Labels.shape) > 1 else 1
    num_patches = int(np.floor((total_samples - Patch_Size) / Stride)) + 1
    adjusted_stride_samples = (
        total_samples - Patch_Size) / (num_patches - 1) if num_patches > 1 else Stride
    adjusted_stride_samples = int(round(adjusted_stride_samples))
    Features_Patch = np.zeros((num_patches, Patch_Size, num_features))
    Labels_Patch = np.zeros((num_patches, Patch_Size, num_labels))
    for i in range(num_patches):
        start_idx = i * adjusted_stride_samples
        end_idx = start_idx + Patch_Size
        if end_idx > total_samples:
            start_idx = total_samples - Patch_Size
            end_idx = total_samples
        Features_Patch[i] = Features[start_idx:end_idx, :]
        Labels_Patch[i] = Labels[start_idx:end_idx,
                                 :] if num_labels > 1 else Labels[start_idx:end_idx]

    return Features_Patch, Labels_Patch


def create_patches_no_labels(Features, Patch_Size, Stride):
    """
    Create overlapping patches from Features for ANN training.

    Parameters
    ----------
    Features : numpy.ndarray
        Input feature matrix of size (Total_Samples, Num_Features).
    Patch_Size : int
        The number of samples for each patch.
    Stride : int
        The number of samples to stride between patches.

    Raises
    ------
    ValueError
        Checks input errors.

    Returns
    -------
    Features_Patch : numpy.ndarray
        Patches of features of size (Num_Patches, Patch_Size, Num_Features).

    """
    total_samples = Features.shape[0]
    num_features = Features.shape[1]

    num_patches = int(np.floor((total_samples - Patch_Size) / Stride)) + 1
    adjusted_stride_samples = (
        total_samples - Patch_Size) / (num_patches - 1) if num_patches > 1 else Stride
    adjusted_stride_samples = int(round(adjusted_stride_samples))

    Features_Patch = np.zeros((num_patches, Patch_Size, num_features))

    for i in range(num_patches):
        start_idx = i * adjusted_stride_samples
        end_idx = start_idx + Patch_Size
        if end_idx > total_samples:
            start_idx = total_samples - Patch_Size
            end_idx = total_samples
        Features_Patch[i] = Features[start_idx:end_idx, :]

    return Features_Patch


def process_dataset(data, patch_size, stride):
    """
    Process the dataset by extracting features and labels, creating patches, and aggregating them.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array where each row corresponds to a sample.
        The features are expected to be in columns 1 to 4, and labels in column 5.
    patch_size : int
        The number of samples in each patch.
    stride : int
        The number of samples to skip between the starts of consecutive patches.

    Returns
    -------
    all_features : numpy.ndarray
        Concatenated array of all feature patches with shape (total_patches, patch_size, num_features).
    all_labels : numpy.ndarray
        Concatenated array of all label patches with shape (total_patches, patch_size).

    Notes
    -----
    - This function iterates over each sample in the dataset, extracts the relevant features and labels,
      and uses `create_patches` to generate patches from them.
    - The patches from all samples are concatenated to form a single array of features and labels.
    """
    all_features = []
    all_labels = []
    for i in range(data.shape[0]):
        features = np.stack(data[i, 1:5], axis=-1)
        labels = data[i, 5]
        features_patches, labels_patches = create_patches(
            features, labels, patch_size, stride)
        all_features.append(features_patches)
        all_labels.append(labels_patches)
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_features, all_labels


def process_dataset_no_labels(data, patch_size, stride):
    """
    Process the dataset by extracting features, creating patches, and aggregating them.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array where each row corresponds to a sample.
        The features are expected to be in columns 1 to 4.
    patch_size : int
        The number of samples in each patch.
    stride : int
        The number of samples to skip between the starts of consecutive patches.

    Returns
    -------
    all_features : numpy.ndarray
        Concatenated array of all feature patches with shape (total_patches, patch_size, num_features).

    Notes
    -----
    - This function iterates over each sample in the dataset, extracts the relevant features,
      and uses `create_patches_no_labels` to generate patches from them.
    - The patches from all samples are concatenated to form a single array of features.
    """
    all_features = []

    for i in range(data.shape[0]):
        features = np.stack(data[i, 1:5], axis=-1)
        features_patches = create_patches_no_labels(
            features, patch_size, stride)
        all_features.append(features_patches)

    all_features = np.concatenate(all_features, axis=0)
    return all_features


def reconstruct_patches(predictions, original_length, patch_size, stride):
    """
    Reconstruct the original sequence from overlapping prediction patches.

    Parameters
    ----------
    predictions : numpy.ndarray
        Predicted patches with shape (num_features, patch_size, num_patches).
    original_length : int
        The length of the original sequence before it was patched.
    patch_size : int
        The number of samples in each patch.
    stride : int
        The number of samples to skip between the starts of consecutive patches.

    Returns
    -------
    reconstructed : numpy.ndarray
        The reconstructed sequence with shape (original_length, num_features).

    Notes
    -----
    - This function reassembles the sequence by overlapping the prediction patches
      and averaging the overlapping regions to smooth transitions.
    - It handles cases where the patches do not perfectly align with the end of the sequence.
    """
    reconstructed = np.zeros((original_length, predictions.shape[0]))
    overlap_count = np.zeros(original_length)
    num_patches = predictions.shape[2]
    for i in range(num_patches):
        start_idx = i * stride
        end_idx = min(start_idx + patch_size, original_length)
        reconstructed[start_idx:end_idx] += predictions[:,
                                                        :, i].T[:end_idx - start_idx]
        overlap_count[start_idx:end_idx] += 1
    reconstructed /= np.maximum(overlap_count[:, None], 1)
    return reconstructed


def reconstruct_original_data(patched_data, original_lengths, patch_size, stride):
    """
    Reconstruct original sequences from patched data for multiple samples.

    Parameters
    ----------
    patched_data : numpy.ndarray
        Array of patched data with shape (num_patches, patch_size, num_features).
    original_lengths : list of int
        List containing the original lengths of each sequence before patching.
    patch_size : int
        The number of samples in each patch.
    stride : int
        The number of samples to skip between the starts of consecutive patches.

    Returns
    -------
    reconstructed_data : list of numpy.ndarray
        List where each element is a reconstructed sequence corresponding to an original sample.

    Notes
    -----
    - This function sequentially reconstructs each original sequence by overlapping its patches
      and averaging the overlapping regions.
    - It accounts for sequences of varying lengths and ensures that each is reconstructed accurately.
    """
    reconstructed_data = []
    current_idx = 0

    for original_length in original_lengths:
        # Initialize arrays to hold the reconstructed sequence and overlap count
        reconstructed = np.zeros((original_length, patched_data.shape[-1]))
        overlap_count = np.zeros(original_length)

        num_patches = int(
            np.floor((original_length - patch_size) / stride)) + 1
        adjusted_stride_samples = (
            (original_length - patch_size) / (num_patches - 1)
            if num_patches > 1 else stride
        )
        adjusted_stride_samples = int(round(adjusted_stride_samples))

        # Iterate over patches and reconstruct the sequence
        for i in range(num_patches):
            start_idx = i * adjusted_stride_samples
            end_idx = min(start_idx + patch_size, original_length)

            reconstructed[start_idx:end_idx] += patched_data[current_idx,
                                                             :end_idx - start_idx, :]
            overlap_count[start_idx:end_idx] += 1

            current_idx += 1

        # Average the overlapping regions
        reconstructed /= np.maximum(overlap_count[:, None], 1)
        reconstructed_data.append(reconstructed)

    return reconstructed_data


def reverse_one_hot_encoding(one_hot_encoded_data, desired_order=[0, 1, 2, 3]):
    """
    Convert one-hot encoded data back to label indices based on a specified label order.

    Parameters
    ----------
    one_hot_encoded_data : numpy.ndarray
        One-hot encoded data of shape (num_samples, num_classes).
    desired_order : list, optional
        List representing the label indices corresponding to the one-hot encoding columns.
        Default is [0, 1, 2, 3].

    Returns
    -------
    labels : numpy.ndarray
        Array of decoded labels with shape (num_samples,).

    Notes
    -----
    - The function uses `np.argmax` to find the index of the maximum value in each row,
      which corresponds to the class label.
    - The `desired_order` parameter maps the indices back to the original labels if they were
      encoded in a specific order.
    """
    label_indices = np.argmax(one_hot_encoded_data, axis=1)
    labels = np.array([desired_order[idx] for idx in label_indices])
    return labels


def max_temporal_modelling(seq, num_states=4):
    """
    Enforce temporal consistency in a sequence by restricting invalid state transitions.

    Parameters
    ----------
    seq : numpy.ndarray or list
        Sequence of state labels (integers) to be processed.
    num_states : int, optional
        Total number of possible states in the sequence. Default is 4.

    Returns
    -------
    seq : numpy.ndarray or list
        The modified sequence with enforced temporal constraints.

    Notes
    -----
    - The function modifies the sequence in-place to ensure that each state transition
      is either to the same state or to the next state in a cyclical manner.
    - If a state transition does not meet this condition, the state at time `t` is set
      to the state at time `t-1`.
    - This is useful for modeling processes where states should progress sequentially.
    """
    for t in range(1, len(seq)):
        if seq[t] != seq[t - 1] and seq[t] != ((seq[t - 1] + 1) % num_states):
            seq[t] = seq[t - 1]
    return seq
