import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

# Load the dataset
data = pd.read_csv('E:\VU\VU jaar 1\MQS\\full_dataset_with_features.csv')

# Extract features and labels
features = data.drop(columns=['id', 'genre'])
labels = data['genre'].astype('category').cat.codes
ids = data['id']

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_array = np.array(features_scaled)
labels_array = np.array(labels)

# Create sequences
window_size = 10

def create_sequences(features, labels, window_size):
    sequences = []
    for i in range(len(features) - window_size + 1):
        seq_features = features[i:i + window_size]
        seq_label = labels[i + window_size - 1]
        sequences.append((seq_features, seq_label))
    return sequences

# Implementing leave-one-out cross-validation manually
group_kfold = GroupKFold(n_splits=len(data['id'].unique()))
results = []

for train_index, test_index in group_kfold.split(features_array, labels_array, groups=ids):
    train_features, test_features = features_array[train_index], features_array[test_index]
    train_labels, test_labels = labels_array[train_index], labels_array[test_index]
    
    train_sequences = create_sequences(train_features, train_labels, window_size)
    test_sequences = create_sequences(test_features, test_labels, window_size)
    
    results.append((train_sequences, test_sequences))

# Define the LSTM model
def create_lstm_model(input_shape, output_size, lstm_units, learning_rate):
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=input_shape, return_sequences=False))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Training and evaluation function
def train_and_evaluate(train_sequences, test_sequences, input_shape, output_size, lstm_units, learning_rate, num_epochs=10, batch_size=32):
    train_x = np.array([seq[0] for seq in train_sequences])
    train_y = np.array([seq[1] for seq in train_sequences])
    test_x = np.array([seq[0] for seq in test_sequences])
    test_y = np.array([seq[1] for seq in test_sequences])

    model = create_lstm_model(input_shape, output_size, lstm_units, learning_rate)
    
    model.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size, verbose=0)
    
    loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
    return accuracy

# Objective function for scipy.optimize.differential_evolution
def objective(params):
    lstm_units, learning_rate, batch_size = params
    lstm_units = int(lstm_units)
    learning_rate = float(learning_rate)
    batch_size = int(batch_size)

    accuracies = []

    for train_sequences, test_sequences in results:
        accuracy = train_and_evaluate(train_sequences, test_sequences, input_shape, output_size, lstm_units, learning_rate, batch_size=batch_size)
        accuracies.append(accuracy)

    # Return the negative average accuracy as the objective value to be minimized
    return -np.mean(accuracies)

# Set parameters
input_shape = (window_size, features_array.shape[1])
output_size = len(data['genre'].unique())

# Define bounds for the hyperparameters
bounds = [(10, 100), (1e-5, 1e-2), (16, 64)]

# Perform hyperparameter optimization using scipy.optimize.differential_evolution
result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=50, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7)

# Get the best parameters
best_params = result.x
best_lstm_units = int(best_params[0])
best_learning_rate = float(best_params[1])
best_batch_size = int(best_params[2])

print(f'Best LSTM Units: {best_lstm_units}')
print(f'Best Learning Rate: {best_learning_rate}')
print(f'Best Batch Size: {best_batch_size}')
