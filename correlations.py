import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = viv_df_1

# Encode the 'genre' column
df['genre_encoded'] = df['genre'].astype('category').cat.codes

# Select only numeric columns for correlation calculation
numeric_columns = df.select_dtypes(include='number').columns

# Calculate Pearson and Spearman correlations
pearson_corr_matrix = df[numeric_columns].corr(method='pearson')
spearman_corr_matrix = df[numeric_columns].corr(method='spearman')

# Plot the Pearson correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(pearson_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Pearson Correlation Matrix')
plt.show()

# Plot the Spearman correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(spearman_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Spearman Correlation Matrix')
plt.show()

# Print the correlation matrices
print("Pearson Correlation Matrix:\n", pearson_corr_matrix)
print("\nSpearman Correlation Matrix:\n", spearman_corr_matrix)
