import pandas as pd

# data = pd.read_csv("Data/Output/cleaned_data_tim_run1.csv")
# deta = pd.read_csv("Data/merged_data_tim_run1.csv")
# print(data.shape, deta.shape)
# percent_missing = data.isnull().sum() * 100 / len(data)
# print(percent_missing)

def linear_interpolate_and_save(data_path, output_path):
    """
    Perform linear interpolation on a dataset and save the interpolated data to a CSV file.

    Parameters:
    data_path (str): Path to the input CSV file.
    output_path (str): Path to save the output interpolated CSV file.
    """
    # Load the dataset
    data = pd.read_csv(data_path)

    # Check for missing values before interpolation
    print("Missing values before interpolation:")
    print(data.isna().mean())

    # Perform linear interpolation
    data_interpolated = data.copy()
    data_interpolated = data_interpolated.interpolate(method='linear', limit_direction='forward', axis=0)

    # Check for missing values after interpolation
    print("Missing values after interpolation:")
    print(data_interpolated.isna().mean())

    # Save the interpolated data
    data_interpolated.to_csv(output_path, index=False)
    print(f"Interpolated data saved to {output_path}")

    # Display the data with interpolated values
    print("Data after linear interpolation:")
    print(data_interpolated.head())

# Example usage
data_path = "E:\VU\VU jaar 1\MQS\\normality_check.csv"
output_path = 'E:\VU\VU jaar 1\MQS\\normality_check_2.csv'
linear_interpolate_and_save(data_path, output_path)
