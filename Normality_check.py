import pandas as pd
import numpy as np
from scipy.stats import shapiro, kstest, anderson
import scipy.stats as stats
import matplotlib.pyplot as plt

data = pd.read_csv("E:\VU\VU jaar 1\MQS\\full_dataset_with_features.csv")

numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('id')
numeric_features.remove('Unnamed: 0')
print(numeric_features)

def check_normality(data, column):
    results = {}
    col_data = data[column]
    # Histogram
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    data[column].hist(bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title('Histogram')

    # Q-Q Plot
    plt.subplot(1, 2, 2)
    stats.probplot(data[column], dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.show()

    # # Shapiro-Wilk Test
    # stat, p_shapiro = shapiro(data[column])
    # results['Shapiro-Wilk'] = (stat, p_shapiro)

    # Anderson-Darling Test
    anderson_result = anderson(col_data, dist='norm')
    results['Anderson-Darling'] = anderson_result
    
    # Kolmogorov-Smirnov Test
    stat, p_kstest = kstest(data[column], 'norm', args=(data[column].mean(), data[column].std()))
    results['Kolmogorov-Smirnov'] = (stat, p_kstest)
    
    # Skewness and Kurtosis
    skewness = data[column].skew()
    kurtosis = data[column].kurt()
    results['Skewness'] = skewness
    results['Kurtosis'] = kurtosis

    #print(f'{column} - Shapiro-Wilk p-value: {p_shapiro}, Kolmogorov-Smirnov p-value: {p_kstest}')
    print(f'Skewness: {skewness}, Kurtosis: {kurtosis}')
    print(f'Anderson-Darling statistic: {anderson_result.statistic}, critical values: {anderson_result.critical_values}')
    return results

# Apply the normality check to all numeric features
normality_results = {}
for feature in numeric_features:
    normality_results[feature] = check_normality(data, feature)
