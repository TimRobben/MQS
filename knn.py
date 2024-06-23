import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

def load_and_split_data(file_path, feature_list, n_splits=5, random_state=42):
    """
    Load the dataset, select specified features, separate features and target, 
    and split into training and testing sets using stratified group k-fold cross-validation 
    to ensure that each group is in the test set once.
    
    Parameters:
    file_path (str): Path to the CSV file
    feature_list (list): List of features to include in the train/test split
    n_splits (int): Number of splits for cross-validation
    random_state (int): Random seed for reproducibility
    
    Returns:
    generator: A generator yielding train-test splits (X_train, X_test, y_train, y_test)
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Separate features and target
    X = data.drop(columns=['id', 'genre'])
    y = data['genre']
    groups = data['id']
    
    # Stratified group k-fold to maintain subject distribution
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for train_idx, test_idx in sgkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        yield X_train, X_test, y_train, y_test

def train_evaluate_knn_cv(file_path, feature_list, n_neighbors=5, n_splits=5):
    """
    Train and evaluate a k-NN classifier using cross-validation with group splits and hyperparameter optimization.
    
    Parameters:
    file_path (str): Path to the CSV file
    feature_list (list): List of features to include in the train/test split
    n_neighbors (int): Number of neighbors for k-NN
    n_splits (int): Number of splits for cross-validation
    
    Returns:
    dict: A dictionary containing average accuracy and classification report
    """
    accuracies = []
    classification_reports = []
    confusion_matrices = []

    for X_train, X_test, y_train, y_test in load_and_split_data(file_path, feature_list, n_splits=n_splits):
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
        knn_classification_report = classification_report(y_test, y_pred_knn, target_names=np.unique(y_train), output_dict=True)
        knn_confusion_matrix = confusion_matrix(y_test, y_pred_knn, labels=np.unique(y_train))

        accuracies.append(knn_accuracy)
        classification_reports.append(knn_classification_report)
        confusion_matrices.append(knn_confusion_matrix)

    avg_accuracy = np.mean(accuracies)

    # Function to compute the mean of nested dictionaries
    def mean_nested_dicts(dicts):
        mean_dict = {}
        for key in dicts[0].keys():
            if isinstance(dicts[0][key], dict):
                mean_dict[key] = mean_nested_dicts([d[key] for d in dicts])
            else:
                mean_dict[key] = np.mean([d[key] for d in dicts])
        return mean_dict

    avg_classification_report = mean_nested_dicts(classification_reports)
    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)

    return {
        'accuracy': avg_accuracy,
        'classification_report': avg_classification_report,
        'confusion_matrix': avg_confusion_matrix
    }

def plot_confusion_matrix(cm, class_names):
    """
    Plot a confusion matrix.
    
    Parameters:
    cm (array): Confusion matrix
    class_names (list): List of class names
    """
    cmd = ConfusionMatrixDisplay(cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    cmd.plot(ax=ax, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

# Example usage
file_path = 'E:/VU/VU jaar 1/MQS/full_dataset_with_features.csv'
feature_list = ['height_mean', 'Y_acc_std', 'Y_acc_mean', 'height_std']

# Train and evaluate the k-NN model using 5-fold cross-validation
n_neighbors = 3
cv_results = train_evaluate_knn_cv(file_path, feature_list, n_neighbors=n_neighbors, n_splits=5)

print(f"Average Accuracy: {cv_results['accuracy']:.4f}")
print("Average Classification Report:")
for class_name, metrics in cv_results['classification_report'].items():
    print(f"{class_name}: {metrics}")

# Plot the average confusion matrix
plot_confusion_matrix(cv_results['confusion_matrix'], np.unique(pd.read_csv(file_path)['genre']))
