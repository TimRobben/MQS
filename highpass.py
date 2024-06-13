import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, periodogram
import matplotlib.pyplot as plt
import os

def highpass_filter(data, cutoff, fs, order=5):
    """
    Apply a high-pass filter to the data.

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
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
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
    plt.plot(filtered_data, label='Filtered', linestyle='--', color='orange')
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel(feature)
    plt.legend()
    plt.show()

def apply_highpass_filter_to_folder(input_folder, output_folder, cutoff, fs, order=5):
    """
    Apply a high-pass filter to all CSV files in the input folder and save the filtered data to the output folder.

    Parameters:
    input_folder (str): Path to the input folder containing CSV files.
    output_folder (str): Path to the output folder to save the filtered CSV files.
    cutoff (float): The cutoff frequency of the filter in Hz.
    fs (float): The sampling rate of the data in Hz.
    order (int): The order of the filter.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.csv', '_hp.csv'))

            # Load the dataset
            data = pd.read_csv(input_path)

            # Identify columns to process (excluding 'Unnamed: 0', 'id', 'Time (s)', and 'genre')
            columns_to_process = [col for col in data.columns if col not in ['Unnamed: 0', 'id', 'Time (s)', 'genre']]

            # Apply high-pass filter to each selected feature
            for feature in columns_to_process:
                filtered_data = highpass_filter(data[feature].values, cutoff, fs, order)
                
                # Visualize the original and filtered data
                visualize_filtered_data(data[feature].values, filtered_data, feature, f'{feature} - Original vs. Filtered')

                # Store the filtered data back to the DataFrame
                data[f'filtered_{feature}'] = filtered_data

            # Save the filtered data
            data.to_csv(output_path, index=False)
            print(f"Filtered data saved to {output_path}")

# Example usage
input_folder = 'data'
output_folder = 'data/output'
cutoff = 2.5  # Cutoff frequency in Hz
fs = 50  # Sampling rate in Hz

apply_highpass_filter_to_folder(input_folder, output_folder, cutoff, fs, order=5)
