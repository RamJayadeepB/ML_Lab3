import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def load_dataset(excel_file):
   
    df = pd.read_excel(excel_file)
    X = df.iloc[:, :-1].values  # all columns except last are features
    y = df.iloc[:, -1].values   # last column is label
    return X, y

def compute_class_statistics(X, y):
    """
    Compute centroids, spreads (standard deviations) for each class,
    and interclass centroid distance.
    """
    X_class_0 = X[y == 0]
    X_class_1 = X[y == 1]
    centroid_0 = np.mean(X_class_0, axis=0)
    centroid_1 = np.mean(X_class_1, axis=0)
    spread_0 = np.std(X_class_0, axis=0)
    spread_1 = np.std(X_class_1, axis=0)
    interclass_distance = np.linalg.norm(centroid_0 - centroid_1)
    return centroid_0, centroid_1, spread_0, spread_1, interclass_distance

def plot_feature_histogram(X, feature_index=0):
    """
    Plot histogram of the specified feature (over all samples).
    Return mean and variance of the feature.
    """
    feature_data = X[:, feature_index]
    plt.figure(figsize=(8,5))
    plt.hist(feature_data, bins=10, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of Feature {feature_index + 1}')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    mean_val = np.mean(feature_data)
    variance_val = np.var(feature_data)
    return mean_val, variance_val

def compute_minkowski_distances(vec1, vec2, max_r=10):
    """
    Compute Minkowski distances between vec1 and vec2 for r=1 to max_r.
    Plot distances vs r.
    Returns list of distances.
    """
    distances = [np.linalg.norm(vec1 - vec2, ord=r) for r in range(1, max_r + 1)]
    plt.figure(figsize=(8,5))
    plt.plot(range(1, max_r + 1), distances, 'o-', color='purple')
    plt.title(f'Minkowski Distances Between Two Samples (r=1 to {max_r})')
    plt.xlabel('r (Minkowski Order)')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.show()
    return distances

def split_train_test(X, y, test_size=0.3, random_state=42):
    """
    Split dataset into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_knn(X_train, y_train, k=3):
    """
    Train k-NN classifier on training data with specified k.
    Returns trained model.
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

def evaluate_accuracy(model, X_test, y_test):
    """
    Return accuracy of the model on test data.
    """
    return model.score(X_test, y_test)

def predict_labels(model, X_test):
    """
    Predict class labels for test data.
    """
    return model.predict(X_test)

def plot_accuracy_vs_k(X_train, X_test, y_train, y_test, max_k=11):
    """
    Compute and plot test set accuracy of k-NN for k=1 to max_k.
    Returns list of accuracies.
    """
    accuracies = []
    for k in range(1, max_k + 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        accuracies.append(acc)
    plt.figure(figsize=(8,5))
    plt.plot(range(1, max_k + 1), accuracies, 'o-', color='green')
    plt.title('k-NN Classifier Accuracy on Test Data vs k')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Test Set Accuracy')
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    plt.show()
    return accuracies

def calculate_performance_metrics(y_test, y_pred):
    """
    Calculate confusion matrix, precision, recall, and F1-score.
    Returns them as tuple.
    """
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return conf_matrix, precision, recall, f1


if __name__ == "__main__":
    ### Load data from Excel file
    X, y = load_dataset(r"C:\Users\bramj\OneDrive\Desktop\classification_dataset.xlsx")


    ### A1: Class separation statistics
    centroid_0, centroid_1, spread_0, spread_1, interclass_dist = compute_class_statistics(X, y)
    print("=== A1: Class Separation Analysis ===")
    print("Centroid Class 0:", centroid_0)
    print("Centroid Class 1:", centroid_1)
    print("Spread (Std Dev) Class 0:", spread_0)
    print("Spread (Std Dev) Class 1:", spread_1)
    print(f"Interclass Centroid Distance: {interclass_dist:.3f}\n")

    ### A2: Feature histogram, mean & variance for feature 1 (index 0)
    mean_feat, var_feat = plot_feature_histogram(X, feature_index=0)
    print(f"=== A2: Feature 1 Histogram ===")
    print(f"Mean of Feature 1: {mean_feat:.3f}")
    print(f"Variance of Feature 1: {var_feat:.3f}\n")

    ### A3: Minkowski distances between first two samples (index 0 and 1)
    minkowski_dist = compute_minkowski_distances(X[0], X[1])
    print("=== A3: Minkowski Distances (r=1 to 10) between first two samples ===")
    print(["{:.3f}".format(d) for d in minkowski_dist], "\n")

    ### A4: Split dataset into train and test subsets
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print("=== A4: Dataset split ===")
    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples\n")

    ### A5: Train k-NN classifier with k=3
    knn_model = train_knn(X_train, y_train, k=3)
    print("=== A5: k-NN classifier trained with k=3 ===\n")

    ### A6: Test accuracy of the k=3 model
    test_acc = evaluate_accuracy(knn_model, X_test, y_test)
    print(f"=== A6: Test Accuracy with k=3: {test_acc:.3f}\n")

    ### A7: Predictions on test set for k=3
    predictions = predict_labels(knn_model, X_test)
    print("=== A7: Predictions on Test Set with k=3 ===")
    print(predictions, "\n")

    ### A8: Plot accuracy vs k from 1 to 11
    accuracy_list = plot_accuracy_vs_k(X_train, X_test, y_train, y_test, max_k=11)
    print("=== A8: Test Accuracy for k=1 to 11 ===")
    print(["{:.3f}".format(acc) for acc in accuracy_list], "\n")

    ### A9: Confusion matrix and performance metrics for k=3
    conf_matrix, precision, recall, f1 = calculate_performance_metrics(y_test, predictions)
    print("=== A9: Confusion Matrix and Metrics for k=3 ===")
    print("Confusion Matrix:\n", conf_matrix)
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}\n")
