import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, periodogram
import matplotlib.pyplot as plt

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

def plot_frequency_spectrum(data, fs, feature):
    """
    Plot the frequency spectrum of the data.

    Parameters:
    data (array-like): The input data.
    fs (float): The sampling rate of the data in Hz.
    feature (str): The feature name for labeling the plot.
    """
    f, Pxx_den = periodogram(data, fs)
    plt.figure(figsize=(12, 6))
    plt.semilogy(f, Pxx_den)
    plt.title(f'Frequency Spectrum of {feature}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()

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

def apply_highpass_filter(data_path, output_path, cutoff, fs, order=5):
    """
    Apply a high-pass filter to the dataset and save the filtered data.

    Parameters:
    data_path (str): Path to the input CSV file.
    output_path (str): Path to save the output filtered CSV file.
    cutoff (float): The cutoff frequency of the filter in Hz.
    fs (float): The sampling rate of the data in Hz.
    order (int): The order of the filter.
    """
    # Load the dataset
    data = pd.read_csv(data_path)

    # Handle missing values by forward filling
    data.fillna(method='ffill', inplace=True)

    # Identify columns to process (excluding 'Unnamed: 0', 'id', 'Time (s)', and 'genre')
    columns_to_process = [col for col in data.columns if col not in ['Unnamed: 0', 'id', 'Time (s)', 'genre']]

    # Apply high-pass filter to each selected feature
    for feature in columns_to_process:
        # Plot the frequency spectrum
        #plot_frequency_spectrum(data[feature].values, fs, feature)
        
        filtered_data = highpass_filter(data[feature].values, cutoff, fs, order)
        
        # Visualize the original and filtered data
        visualize_filtered_data(data[feature].values, filtered_data, feature, f'{feature} - Original vs. Filtered')

        # Store the filtered data back to the DataFrame
        data[f'filtered_{feature}'] = filtered_data

    # Save the filtered data
    data.to_csv(output_path, index=False)
    print(f"Filtered data saved to {output_path}")

    # Display the filtered data
    print("Data after applying high-pass filter:")
    print(data.head())

# Example usage
data_path = 'Data/Full_data/full_orig_data/full_dataset_tim_run_1.csv'
output_path = 'data/full_data/hp_data/filtered_data_highpass.csv'
cutoff = 2.5  # Cutoff frequency in Hz
fs = 50  # Sampling rate in Hz

apply_highpass_filter(data_path, output_path, cutoff, fs, order=5)
