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

# Define custom metrics
def precision_m(y_true, y_pred):
    y_true = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1]), tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def recall_m(y_true, y_pred):
    y_true = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1]), tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

# Define the LSTM model
def create_lstm_model(input_shape, output_size):
    model = Sequential()
    model.add(LSTM(65, input_shape=input_shape, return_sequences=False))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0056), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy', precision_m, recall_m, f1_m])
    return model

# Training and evaluation function
def train_and_evaluate(train_sequences, test_sequences, input_shape, output_size, num_epochs=10, batch_size=65):
    train_x = np.array([seq[0] for seq in train_sequences])
    train_y = np.array([seq[1] for seq in train_sequences])
    test_x = np.array([seq[0] for seq in test_sequences])
    test_y = np.array([seq[1] for seq in test_sequences])

    model = create_lstm_model(input_shape, output_size)
    
    model.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size, verbose=0)
    
    loss, accuracy, precision, recall, f1 = model.evaluate(test_x, test_y, verbose=0)
    return accuracy, precision, recall, f1

# Set parameters
input_shape = (window_size, features_array.shape[1])
output_size = len(data['genre'].unique())

# Perform leave-one-out cross-validation
accuracies = []
precisions = []
recalls = []
f1_scores = []

for train_sequences, test_sequences in results:
    accuracy, precision, recall, f1 = train_and_evaluate(train_sequences, test_sequences, input_shape, output_size)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Display the average metrics
average_accuracy = np.mean(accuracies)
average_precision = np.mean(precisions)
average_recall = np.mean(recalls)
average_f1 = np.mean(f1_scores)

print(f'Average Accuracy: {average_accuracy, accuracies}')
print(f'Average Precision: {average_precision, precisions}')
print(f'Average Recall: {average_recall, recalls}')
print(f'Average F1 Score: {average_f1, f1_scores}')
