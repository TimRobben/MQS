import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

def apply_lof_and_plot(data_path, output_dir, n_neighbors=20, remove=0):
    # Load the dataset
    data = pd.read_csv(data_path)

    # Identify numeric features
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()

    # Remove irrelevant features
    for col in ['id', 'Time (s)']:
        if col in numeric_features:
            numeric_features.remove(col)

    # Function to apply Local Outlier Factor (LOF)
    def apply_lof(data, feature, n_neighbors=25):
        feature_data = data[[feature]].dropna()
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        outliers = lof.fit_predict(feature_data)
        outlier_series = pd.Series(outliers, index=feature_data.index)
        data[f'outlier_lof_{feature}'] = outlier_series
        return data

    # Function to plot LOF outliers
    def plot_lof_outliers(data, feature, output_dir='.'):
        feature_data = data[[feature]].dropna()
        outliers = data[f'outlier_lof_{feature}'].dropna()
        
        plt.figure(figsize=(10, 6))
        plt.hist(feature_data, bins=30, alpha=0.6, label='Data')
        plt.scatter(feature_data[outliers == -1], np.zeros_like(feature_data[outliers == -1]), color='red', label='Outliers', marker='x')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.title(f'LOF Outliers for {feature}')
        plt.legend()
        plt.savefig(f"{output_dir}/LOF_{feature}.png")
        plt.close()

    # Apply LOF to all numeric features and store the results
    for feature in numeric_features:
        data = apply_lof(data, feature)
        #plot_lof_outliers(data, feature, output_dir)

    # Replace outliers with NaN if the remove parameter is set to 1
    if remove == 1:
        for feature in numeric_features:
            data.loc[data[f'outlier_lof_{feature}'] == -1, feature] = np.nan

        # Drop the outlier flag columns
        outlier_columns = [f'outlier_lof_{feature}' for feature in numeric_features]
        data.drop(columns=outlier_columns, inplace=True)

    # Display the data after processing
    print("Data after applying LOF:")
    print(data.head())

    # Save the cleaned data if outliers were replaced with NaN
    if remove == 1:
        cleaned_data_path = f"{output_dir}/cleaned_data_tim_run1.csv"
        data.to_csv(cleaned_data_path, index=False)
        print(f"Cleaned data saved to {cleaned_data_path}")

# Example usage
data_path = 'Data/merged_data_tim_run1.csv'
output_dir = 'Data/output'  # Directory to save the plots
apply_lof_and_plot(data_path, output_dir, n_neighbors=20, remove=1)
