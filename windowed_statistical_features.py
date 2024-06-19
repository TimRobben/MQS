original_data = df

# Define the time window size (30 seconds) and minimum data points per window
window_size = 30  # in seconds
min_data_points = 10

# Initialize a list to store the calculated features
all_features = []

# Process each genre separately
genres = original_data['genre'].unique()
for genre in genres:
    genre_data = original_data[original_data['genre'] == genre].reset_index(drop=True)

    # Process each ID within the current genre
    ids = genre_data['id'].unique()
    for id_ in ids:
        id_genre_data = genre_data[genre_data['id'] == id_].reset_index(drop=True)

        # Calculate the number of 30-second windows for the current ID and genre
        num_windows = len(id_genre_data) // window_size

        # Iterate over each 30-second window for the current ID and genre
        for i in range(num_windows):
            # Calculate the start and end indices for the current window
            start_idx = i * window_size
            end_idx = (i + 1) * window_size

            # Extract data within the current time window for the current ID and genre
            window_data = id_genre_data.iloc[start_idx:end_idx]

            # Check if the window has enough data points
            if len(window_data) >= min_data_points:
                # Calculate statistical features for each sensor axis
                statistical_features = {
                    'id': id_,
                    'X_acc_mean': window_data['X (m/s^2)_acc'].mean(),
                    'X_acc_std': window_data['X (m/s^2)_acc'].std(),
                    'Y_acc_mean': window_data['Y (m/s^2)_acc'].mean(),
                    'Y_acc_std': window_data['Y (m/s^2)_acc'].std(),
                    'Z_acc_mean': window_data['Z (m/s^2)_acc'].mean(),
                    'Z_acc_std': window_data['Z (m/s^2)_acc'].std(),
                    'X_lin_acc_mean': window_data['X (m/s^2)_lin_acc'].mean(),
                    'X_lin_acc_std': window_data['X (m/s^2)_lin_acc'].std(),
                    'Y_lin_acc_mean': window_data['Y (m/s^2)_lin_acc'].mean(),
                    'Y_lin_acc_std': window_data['Y (m/s^2)_lin_acc'].std(),
                    'Z_lin_acc_mean': window_data['Z (m/s^2)_lin_acc'].mean(),
                    'Z_lin_acc_std': window_data['Z (m/s^2)_lin_acc'].std(),
                    'height_mean': window_data['Height (m)'].mean(),
                    'height_std': window_data['Height (m)'].std(),
                    'height_median': window_data['Height (m)'].median(),
                    'velocity_mean': window_data['Velocity (m/s)'].mean(),
                    'velocity_std': window_data['Velocity (m/s)'].std(),
                    'velocity_median': window_data['Velocity (m/s)'].median(),
                    'genre': genre
                }
                all_features.append(statistical_features)

# Convert the list of dictionaries to a DataFrame
features_df = pd.DataFrame(all_features)

# Display the resulting DataFrame
print(features_df)
