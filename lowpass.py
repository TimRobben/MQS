import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from Percent_missing import percent_missing

def lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply a low-pass filter to the data.

    Parameters:
    data (array-like): The input data to filter.
    cutoff (float): The cutoff frequency of the filter in Hz.
    fs (float): The sampling rate of the data in Hz.
    order (int): The order of the filter.

    Returns:
    filtered_data (array-like): The filtered data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def visualize_filtered_data(original_data, filtered_data, feature, title):
    """
    Visualize the original and filtered data.

    Parameters:
    original_data (array-like): The original data.
    filtered_data (array-like): The filtered data.
    feature (str): The feature name for labeling the plot.
    title (str): The title of the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(original_data, label='Original', alpha=0.75)
    plt.plot(filtered_data, label='Filtered', linestyle='--')
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel(feature)
    plt.legend()
    plt.show()

def apply_lowpass_filter(data_path, output_path, cutoff, fs, order=5):
    """
    Apply a low-pass filter to the dataset and save the filtered data.

    Parameters:
    data_path (str): Path to the input CSV file.
    output_path (str): Path to save the output filtered CSV file.
    cutoff (float): The cutoff frequency of the filter in Hz.
    fs (float): The sampling rate of the data in Hz.
    order (int): The order of the filter.
    """
    # Load the dataset
    data = pd.read_csv(data_path)

    # Identify columns to process (excluding 'Unnamed: 0', 'id', 'Time (s)', and 'genre')
    columns_to_process = [col for col in data.columns if col not in ['Unnamed: 0', 'id', 'Time (s)', 'genre']]
    print(columns_to_process)
    # Apply low-pass filter to each selected feature
    for feature in columns_to_process:
        filtered_data = lowpass_filter(data[feature].values, cutoff, fs, order)
        
        # Visualize the original and filtered data
        visualize_filtered_data(data[feature].values, filtered_data, feature, f'{feature} - Original vs. Filtered')
        
        # Store the filtered data back to the DataFrame
        data[f'filtered_{feature}'] = filtered_data

    # Save the filtered data
    data.to_csv(output_path, index=False)
    print(f"Filtered data saved to {output_path}")
    #percent_missing(output_path)

    # Display the filtered data
    # print("Data after applying low-pass filter:")
    # print(data.head())

# Example usage
data_path = 'Data/Output/interpolated_tim_run_1.csv'
output_path = 'data/output/filtered_data.csv'
cutoff = 2.5  # Cutoff frequency in Hz
fs = 50  # Sampling rate in Hz

apply_lowpass_filter(data_path, output_path, cutoff, fs, order=5)
