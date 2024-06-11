import pandas as pd

# Load your original dataset
original_data = pd.read_csv('original_dataset.csv')

# Load the dataset containing statistical features
features_df = pd.read_csv('statistical_features_30s_windows_by_genre.csv')

# Define the time window size (30 seconds)
window_size = 30  # in seconds

# Initialize lists to store data
ids = []
time_windows = []

# Process each genre separately
for genre in features_df['genre'].unique():
    genre_data = features_df[features_df['genre'] == genre].reset_index(drop=True)
    num_windows = len(genre_data)

    # Generate unique ids for each window within the genre
    genre_ids = original_data[original_data['genre'] == genre]['id'].values[:num_windows]  # Use id from original_data
    ids.extend(genre_ids)

    # Create time windows for each window within the genre
    genre_time_windows = [f'{window_idx * window_size}-{(window_idx + 1) * window_size}' for window_idx in range(num_windows)]
    time_windows.extend(genre_time_windows)

# Add id and time window columns to the features dataframe
features_df['id'] = ids
features_df['time_window'] = time_windows

# Reorder columns
features_df = features_df[['id', 'time_window', 'X_acc_mean', 'X_acc_std', 'Y_acc_mean', 'Y_acc_std', 'Z_acc_mean', 'Z_acc_std', 'X_lin_acc_mean', 'X_lin_acc_std', 'Y_lin_acc_mean', 'Y_lin_acc_std', 'Z_lin_acc_mean', 'Z_lin_acc_std', 'genre']]
