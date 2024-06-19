import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

def load_and_split_data(file_path, test_size=0.2, random_state=42):
    """
    Load the dataset, separate features and target, and split into training and testing sets,
    ensuring that the split maintains the proportion of subjects.
    
    Parameters:
    file_path (str): Path to the CSV file
    test_size (float): Proportion of the dataset to include in the test split
    random_state (int): Random seed for reproducibility
    
    Returns:
    tuple: Training and testing features and labels (X_train, X_test, y_train, y_test)
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    # features = ['height_std', 'Z_lin_acc_std', 'X_acc_std', 'Z_acc_std', 'Agg_WeightedAvgFreq', 
    #             'Agg_SpectralEntropy', 'Agg_TotalAmplitude', 'velocity_std', 'Y_acc_std', 'X_lin_acc_std', 
    #             'velocity_median', 'Agg_MaxAmplitudeFreq', 'velocity_mean', 'Y_acc_mean', 'Y_lin_acc_mean', 
    #             'Y_lin_acc_std', 'Time (s)', 'height_mean', 'X_acc_mean', 'X_lin_acc_mean', 'Z_lin_acc_mean', 
    #             'Z_acc_mean', 'height_median']
    # Separate features and target
    X = data.drop(columns=['id', 'genre'])
    # Select specified features and target
    # X = data[features]
    y = data['genre']
    groups = data['id']
    
    # Stratified group k-fold to maintain subject distribution
    sgkf = StratifiedGroupKFold(n_splits=int(1/test_size), shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    return X_train, X_test, y_train, y_test

def train_evaluate_knn(X_train, X_test, y_train, y_test, n_neighbors=5):
    """
    Train and evaluate a k-NN classifier.
    
    Parameters:
    X_train (pd.DataFrame): Training features
    X_test (pd.DataFrame): Testing features
    y_train (pd.Series): Training labels
    y_test (pd.Series): Testing labels
    n_neighbors (int): Number of neighbors for k-NN
    
    Returns:
    dict: A dictionary containing accuracy, classification report, and predictions
    """
    # Define the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X_train.columns)
        ])
    
    # Define the k-NN model
    knn_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=n_neighbors))
    ])
    
    # Train the k-NN model
    knn_model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred_knn = knn_model.predict(X_test)
    knn_accuracy = accuracy_score(y_test, y_pred_knn)
    knn_classification_report = classification_report(y_test, y_pred_knn, target_names=np.unique(y_train))
    
    return {
        'accuracy': knn_accuracy,
        'classification_report': knn_classification_report,
        'y_pred': y_pred_knn
    }

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot a confusion matrix.
    
    Parameters:
    y_true (pd.Series): True labels
    y_pred (pd.Series): Predicted labels
    class_names (list): List of class names
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cmd = ConfusionMatrixDisplay(cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    cmd.plot(ax=ax, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('Data/Full_data/hp_data/Confusion_matrix.png')
    plt.show()

# Example usage
file_path = 'E:\VU\VU jaar 1\MQS\\filtered_features_df.csv'
X_train, X_test, y_train, y_test = load_and_split_data(file_path)
#neighbors_list = range(1, 21)
neighbors_list = [2]

accuracies = []

for n_neighbors in neighbors_list:
    knn_results = train_evaluate_knn(X_train, X_test, y_train, y_test, n_neighbors=n_neighbors)
    accuracies.append(knn_results['accuracy'])
    print(f"Number of Neighbors: {n_neighbors}")
    print(f"Accuracy: {knn_results['accuracy']:.4f}")
    print(knn_results['classification_report'])
    plot_confusion_matrix(y_test, knn_results['y_pred'], np.unique(y_train))

#Plot accuracy vs. number of neighbors
plt.figure(figsize=(10, 6))
plt.plot(neighbors_list, accuracies, marker='o')
plt.title('Accuracy vs. Number of Neighbors')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
