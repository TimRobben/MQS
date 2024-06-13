import pandas as pd

acc_pop = pd.read_csv('Data/Tim_dance2/Accelerometer.csv', delimiter =',')
acc_rock = pd.read_csv('Data/Tim_rock2/Accelerometer.csv', delimiter =',')
acc_classic = pd.read_csv('Data/Tim_jazz2/Accelerometer.csv', delimiter =',')

lin_acc_pop = pd.read_csv('Data/Tim_dance2/Linear Accelerometer.csv', delimiter =',')
lin_acc_rock = pd.read_csv('Data/Tim_rock2/Linear Accelerometer.csv', delimiter =',')
lin_acc_classic = pd.read_csv('Data/Tim_jazz2/Linear Accelerometer.csv', delimiter =',')

# Merge the individual genre datasets into one dataset per genre

pop_combined_df = pd.merge(acc_pop, lin_acc_pop, on='Time (s)', how='left', suffixes=('_acc', '_lin_acc'))
rock_combined_df = pd.merge(acc_rock, lin_acc_rock, on='Time (s)', how='left', suffixes=('_acc', '_lin_acc'))
classic_combined_df = pd.merge(acc_classic, lin_acc_classic, on='Time (s)', how='left', suffixes=('_acc', '_lin_acc'))

# create genre labels

pop_combined_df['genre'] = 'Dance'
rock_combined_df['genre'] = 'Rock'
classic_combined_df['genre'] = 'Jazz'

# Concatenate the DataFrames
combined_df = pd.concat([pop_combined_df, rock_combined_df, classic_combined_df], ignore_index=True)

# Verify the result
print(combined_df)

# add ID label
combined_df.insert(0, 'id', 1)

# save to csv
combined_df.to_csv('Data/merged_data_tim_run_2.csv')

# @title genre vs X (m/s^2)_acc

from matplotlib import pyplot as plt
import seaborn as sns
figsize = (12, 1.2 * len(combined_df['genre'].unique()))
plt.figure(figsize=figsize)
sns.violinplot(combined_df, x='X (m/s^2)_acc', y='genre', inner='box', palette='Dark2')
sns.despine(top=True, right=True, bottom=True, left=True)
plt.title('X acceleration for each genre for id 1')
plt.show()
