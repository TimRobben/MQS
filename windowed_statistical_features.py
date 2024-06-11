original_data = pd.read_csv('/content/merged_data_viv_run_1.csv', delimiter =',')

# Define the time window size (30 seconds)
window_size = 30  # in seconds

# Initialize a list to store the calculated features
all_features = []

# Process each genre separately
genres = original_data['genre'].unique()
for genre in genres:
    genre_data = original_data[original_data['genre'] == genre].reset_index(drop=True)

    # Calculate the number of 30-second windows for the current genre
    num_windows = len(genre_data) // (window_size * 50)

    # Iterate over each 30-second window for the current genre
    for i in range(num_windows):
        # Calculate the start and end indices for the current window
        start_idx = i * window_size * 50
        end_idx = (i + 1) * window_size * 50

        # Extract data within the current time window for the current genre
        window_data = genre_data.iloc[start_idx:end_idx]

        # Check if the window has enough data points
        if len(window_data) >= 10:  # Adjust as needed based on your dataset and desired statistical features
            # Calculate statistical features for each sensor axis
            statistical_features = {
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
                'genre': genre
            }
            all_features.append(statistical_features)

# Convert the list of dictionaries to a DataFrame
features_df = pd.DataFrame(all_features)
