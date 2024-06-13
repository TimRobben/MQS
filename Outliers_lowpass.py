import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply a low-pass filter to the data.

    Parameters:
    data (array-like): The input data to filter.
    cutoff (float): The cutoff frequency of the filter.
    fs (float): The sampling rate of the data.
    order (int): The order of the filter.

    Returns:
    filtered_data (array-like): The filtered data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def apply_lowpass_filter_and_save(data_path, output_path, cutoff, fs, order=5):
    """
    Apply a low-pass filter to the dataset and save the filtered data to a CSV file.

    Parameters:
    data_path (str): Path to the input CSV file.
    output_path (str): Path to save the output filtered CSV file.
    cutoff (float): The cutoff frequency of the filter.
    fs (float): The sampling rate of the data.
    order (int): The order of the filter.
    """
    # Load the dataset
    data = pd.read_csv(data_path)

    # Identify numeric features
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()

    # Apply low-pass filter to each numeric feature
    for feature in numeric_features:
        data[feature] = lowpass_filter(data[feature].values, cutoff, fs, order)

    # Save the filtered data
    data.to_csv(output_path, index=False)
    print(f"Filtered data saved to {output_path}")

    # Display the data with filtered values
    print("Data after low-pass filtering:")
    print(data.head(), data.shape)


# Example usage
data_path = 'data/merged_data_tim_run1.csv'
output_path = 'Data/output/filtered_data.csv'
cutoff = 0.1  # Cutoff frequency as a fraction of the sampling rate
fs = 1.0  # Sampling rate, adjust based on your data

apply_lowpass_filter_and_save(data_path, output_path, cutoff, fs, order=5)
