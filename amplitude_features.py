import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.stats import entropy

# Load your dataset and rename columns
df = pd.read_csv('/content/dataset_viv_2_run.csv')
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

# Assuming you have a 'genre' column in your dataset
genres = df['genre'].unique()  # List of unique genres
print("Genres:", genres)

fs = 50  # Sampling rate (Hz)
window_size = 30 * fs  # Window size (e.g., 30 seconds)

# compute the frequency features
def compute_frequency_features(data, fs):
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1/fs)[:N//2]
    magnitude = 2.0 / N * np.abs(yf[0:N//2])
    
    # Frequency with highest amplitude
    max_amplitude_freq = xf[np.argmax(magnitude)]
    
    # Total amplitude
    total_amplitude = np.sum(magnitude)
    
    # Frequency weighted signal average
    weighted_avg_freq = np.sum(xf * magnitude) / total_amplitude if total_amplitude != 0 else 0
    
    # Power spectral entropy
    frequencies, psd = welch(data, fs, nperseg=min(256, N))
    normalized_psd = psd / np.sum(psd)
    spectral_entropy = entropy(normalized_psd)
    
    return max_amplitude_freq, total_amplitude, weighted_avg_freq, spectral_entropy

# Initialize lists to store aggregated features
agg_max_amplitude_freq = []
agg_total_amplitude = []
agg_weighted_avg_freq = []
agg_spectral_entropy = []
agg_genre = []

# loop over genres
for genre in genres:
    genre_data = df[df['genre'] == genre]
    num_windows = len(genre_data) // window_size

    # loop over timepoints within a window
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window_data = genre_data['AccX'].values[start:end]
        
        max_amp_freq, total_amp, weighted_avg_freq, spec_entropy = compute_frequency_features(window_data, fs)
        
        agg_max_amplitude_freq.append(max_amp_freq)
        agg_total_amplitude.append(total_amp)
        agg_weighted_avg_freq.append(weighted_avg_freq)
        agg_spectral_entropy.append(spec_entropy)
        agg_genre.append(genre)

# Create DataFrame to store aggregated features
agg_features_df = pd.DataFrame({
    'Genre': agg_genre,
    'Agg_MaxAmplitudeFreq': agg_max_amplitude_freq,
    'Agg_TotalAmplitude': agg_total_amplitude,
    'Agg_WeightedAvgFreq': agg_weighted_avg_freq,
    'Agg_SpectralEntropy': agg_spectral_entropy
})


