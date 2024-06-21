import pandas as pd
import numpy as np
import pandas as pd
import os

def merge_csv_files(directory_path, output_file):
    """
    Merges all CSV files in the specified directory into a single DataFrame and saves it to an output file.

    Parameters:
    directory_path (str): The path to the directory containing the CSV files.
    output_file (str): The path to the output CSV file.

    Returns:
    pd.DataFrame: The merged DataFrame.
    """
    # Initialize an empty list to hold the DataFrames
    dataframes = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join(directory_path, filename)
            
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            
            # Append the DataFrame to the list
            dataframes.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Optionally, save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)

    return merged_df

# Example usage
# directory_path = 'Data/Full_data/full_orig_data/'  # Update with your directory path
# output_file = 'E:\VU\VU jaar 1\MQS\\normality_check.csv'  # Update with your desired output file path
# merged_df = merge_csv_files(directory_path, output_file)

# # Display the merged DataFrame
# print("Merged DataFrame:")
# print(merged_df)

def percent_missing(path):
    data = pd.read_csv(path)
    #deta = pd.read_csv("Data/merged_data_tim_run1.csv")
    #print(data.shape)
    missing = data.isnull().sum()
    #print("missing values:", missing)
    percent_missing = data.isnull().sum() * 100 / len(data)
    print("percentage missing:", percent_missing)

#percent_missing('Data/Full_data/full_orig_data/full_dataset_tim_run_1.csv')

def remove_first_2_columns(path):
    percent_missing(path)
    df = pd.read_csv(path)
    df.drop(df.columns[df.columns.str.contains('^Unnamed')], axis=1, inplace=True)
    if 'filtered_Unnamed: 0.1' in df.columns:
        df.drop(columns=['filtered_Unnamed: 0.1'], inplace=True)
    df.to_csv(path, index=False)
    percent_missing(path)

#remove_first_2_columns('Data/Full_data/full_orig_data/full_dataset_viv_run_2.csv')
def count_instances_per_id(path, id_column):
    df = pd.read_csv(path)
    return print(df[id_column].value_counts())

#count_instances_per_id('E:\VU\VU jaar 1\MQS\\full_dataset_with_features.csv', 'id')