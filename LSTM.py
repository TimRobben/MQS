import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd

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
def create_lstm_model(input_shape, output_size):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, return_sequences=False))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Training and evaluation function
def train_and_evaluate(train_sequences, test_sequences, input_shape, output_size, num_epochs=10, batch_size=32):
    train_x = np.array([seq[0] for seq in train_sequences])
    train_y = np.array([seq[1] for seq in train_sequences])
    test_x = np.array([seq[0] for seq in test_sequences])
    test_y = np.array([seq[1] for seq in test_sequences])

    model = create_lstm_model(input_shape, output_size)
    
    model.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size, verbose=0)
    
    loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
    return accuracy

# Set parameters
input_shape = (window_size, features_array.shape[1])
output_size = len(data['genre'].unique())

# Perform leave-one-out cross-validation
accuracies = []

for train_sequences, test_sequences in results:
    accuracy = train_and_evaluate(train_sequences, test_sequences, input_shape, output_size)
    accuracies.append(accuracy)

# Display the average accuracy
average_accuracy = np.mean(accuracies)
print(f'Average Accuracy: {average_accuracy}')
