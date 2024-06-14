import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.stats import entropy

# Load your dataset and rename columns
df = pd.read_csv('/content/full_dataset_viv_run_1.csv')
df.rename(columns={
    'Time (s)': 'Time',
    'X (m/s^2)_acc': 'AccX',
    'Y (m/s^2)_acc': 'AccY',
    'Z (m/s^2)_acc': 'AccZ',
    'X (m/s^2)_lin_acc': 'LinAccX',
    'Y (m/s^2)_lin_acc': 'LinAccY',
    'Z (m/s^2)_lin_acc': 'LinAccZ',
    'Height (m)': 'Height',
    'Velocity (m/s)': 'Velocity'
}, inplace=True)

# Define parameters
fs = 50  # Sampling rate (Hz)
window_size = 5 * fs  # Window size (e.g., 5 seconds)
overlap = 0.5  # Overlap percentage (e.g., 50% overlap)

# Function to compute frequency domain features
def compute_frequency_features(data, fs):
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1/fs)[:N//2]
    magnitude = 2.0 / N * np.abs(yf[0:N//2])
    
    # 1. Frequency with highest amplitude
    max_amplitude_freq = xf[np.argmax(magnitude)]
    
    # 2. Frequency weighted signal average
    total_amplitude = np.sum(magnitude)
    weighted_avg_freq = np.sum(xf * magnitude) / total_amplitude if total_amplitude != 0 else 0
    
    # 3. Power spectral entropy
    frequencies, psd = welch(data, fs, nperseg=min(256, N))
    normalized_psd = psd / np.sum(psd)
    spectral_entropy = entropy(normalized_psd)
    
    return max_amplitude_freq, weighted_avg_freq, spectral_entropy

# Initialize lists to store features
window_indices = []
max_amplitude_freqs = []
weighted_avg_freqs = []
spectral_entropies = []

# Iterate over time windows
start = 0
while start + window_size <= len(df):
    end = start + window_size
    window_data = df['AccX'].values[start:end]
    
    max_amp_freq, weighted_avg_freq, spec_entropy = compute_frequency_features(window_data, fs)
    
    window_indices.append(start)
    max_amplitude_freqs.append(max_amp_freq)
    weighted_avg_freqs.append(weighted_avg_freq)
    spectral_entropies.append(spec_entropy)
    
    start += int(window_size * (1 - overlap))

# Create DataFrame to store features
features_df = pd.DataFrame({
    'WindowStartIndex': window_indices,
    'MaxAmplitudeFreq': max_amplitude_freqs,
    'WeightedAvgFreq': weighted_avg_freqs,
    'SpectralEntropy': spectral_entropies
})

# Merge features_df with original DataFrame based on time window indices
df = pd.merge_asof(df, features_df, left_index=True, right_on='WindowStartIndex', direction='backward')

# Output or use df with added features
print(df.head())
