"""
Preprocessing functions for EEG signals.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def apply_bandpass_filter(data, lowcut=0.5, highcut=40, fs=173.61, order=5):
    """
    Apply a Butterworth bandpass filter to the EEG data.
    
    Args:
        data (numpy.ndarray): EEG data to filter
        lowcut (float): Lower cutoff frequency in Hz
        highcut (float): Upper cutoff frequency in Hz
        fs (float): Sampling rate in Hz
        order (int): Filter order
        
    Returns:
        numpy.ndarray: Filtered EEG data
    """
    # Design a butterworth bandpass filter
    b, a = butter(order, [lowcut / fs * 2, highcut / fs * 2], btype='bandpass')
    
    # Filter the data
    filtered_data = lfilter(b, a, data)
    
    return filtered_data

def apply_baseline_correction(filtered_data):
    """
    Apply baseline correction to remove DC offset.
    
    Args:
        filtered_data (numpy.ndarray): Filtered EEG data
        
    Returns:
        numpy.ndarray: Baseline corrected EEG data
    """
    # Calculate the mean of the data
    baseline = np.mean(filtered_data)
    
    # Subtract the mean from the data
    corrected_data = filtered_data - baseline
    
    return corrected_data

def normalize_data(data):
    """
    Normalize the data to a common range (-1 to 1).
    
    Args:
        data (numpy.ndarray): EEG data to normalize
        
    Returns:
        numpy.ndarray: Normalized EEG data
    """
    # Find the maximum absolute value
    max_abs = np.max(np.abs(data))
    
    # Normalize the data
    normalized_data = data / max_abs
    
    return normalized_data

def preprocess_eeg_data(data):
    """
    Apply all preprocessing steps to the EEG data:
    1. Bandpass filtering
    2. Baseline correction
    3. Normalization
    
    Args:
        data (numpy.ndarray): Raw EEG data
        
    Returns:
        numpy.ndarray: Preprocessed EEG data
    """
    # 1. Bandpass filtering
    filtered_data = apply_bandpass_filter(data)
    
    # 2. Baseline correction
    corrected_data = apply_baseline_correction(filtered_data)
    
    # 3. Normalization
    normalized_data = normalize_data(corrected_data)
    
    return normalized_data

def plot_eeg_signal(data, title="EEG Signal", stage=None):
    """
    Plot an EEG signal.
    
    Args:
        data (numpy.ndarray): EEG data to plot
        title (str): Plot title
        stage (str, optional): Processing stage description
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    
    if stage:
        plt.title(f"{title} - {stage}")
    else:
        plt.title(title)
        
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()
