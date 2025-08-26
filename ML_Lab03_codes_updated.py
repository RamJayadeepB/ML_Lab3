
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --------------------------
# A1: Intraclass Spread & Interclass Distance
# --------------------------
def class_statistics(X, y, class1, class2):
    X1 = X[y == class1]  # Samples belonging to class1
    X2 = X[y == class2]  # Samples belonging to class2

    centroid1 = X1.mean(axis=0)  # Mean of features for class1
    centroid2 = X2.mean(axis=0)  # Mean of features for class2

    spread1 = X1.std(axis=0)  # Spread (standard deviation) of class1
    spread2 = X2.std(axis=0)  # Spread (standard deviation) of class2

    distance = np.linalg.norm(centroid1 - centroid2)  # Euclidean distance between centroids

    return centroid1, centroid2, spread1, spread2, distance

# --------------------------
# A2: Histogram for one feature
# --------------------------
def feature_histogram(X, feature_index):
    feature = X[:, feature_index]
    hist, bins = np.histogram(feature, bins=10)  # Histogram with 10 bins
    mean = np.mean(feature)
    variance = np.var(feature)
    return hist, bins, mean, variance

# --------------------------
# A3: Minkowski Distance Plot
# --------------------------
def minkowski_distances(vec1, vec2, max_r=10):
    distances = []
    for r in range(1, max_r + 1):
        dist = np.sum(np.abs(vec1 - vec2) ** r) ** (1 / r)
        distances.append(dist)
    return distances

# --------------------------
# A4: Train-Test Split
# --------------------------
def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# --------------------------
# A5: Train kNN Classifier (k=3)
# --------------------------
def train_knn(X_train, y_train, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

# --------------------------
# A6: Accuracy on Test Set
# --------------------------
def evaluate_accuracy(model, X_test, y_test):
    return model.score(X_test, y_test)

# --------------------------
# A7: Predictions
# --------------------------
def predict_samples(model, X_test, n=5):
    return model.predict(X_test[:n])

# --------------------------
# A8: Accuracy vs k
# --------------------------
def knn_accuracy_vs_k(X_train, X_test, y_train, y_test, max_k=11):
    k_values = list(range(1, max_k + 1))
    accuracies = []
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        accuracies.append(acc)
    return k_values, accuracies

# --------------------------
# A9: Confusion Matrix & Metrics
# --------------------------
def evaluate_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return cm, report

# --------------------------
# Main Section
# --------------------------
if __name__ == "__main__":
    # Load dataset (CSV file with 2200 samples)
    df = pd.read_csv(r"C:\Users\DELL\Downloads\Crop_recommendation.csv")
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target_col = 'label'

    X = df[feature_cols].values  # Feature matrix
    y = df[target_col].values    # Target labels

    # A1: Compare two classes (e.g., rice vs maize)
    c1, c2, s1, s2, dist = class_statistics(X, y, 'rice', 'maize')
    print("A1 Centroid Distance (rice vs maize):", dist)

    # A2: Histogram for Nitrogen (N) feature
    hist, bins, mean, var = feature_histogram(X, feature_index=0)
    plt.hist(X[:, 0], bins=10, edgecolor='black')
    plt.title("Histogram for Nitrogen (N)")
    plt.show()
    print("A2 Mean (N):", mean, "Variance (N):", var)

    # A3: Minkowski distances between first two samples
    vec1, vec2 = X[0], X[1]
    distances = minkowski_distances(vec1, vec2)
    plt.plot(range(1, 11), distances, marker='o')
    plt.title("Minkowski Distance (r=1 to 10)")
    plt.xlabel("r")
    plt.ylabel("Distance")
    plt.show()

    # A4: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # A5: Train kNN (k=3)
    model_k3 = train_knn(X_train, y_train, k=3)

    # A6: Test Accuracy
    acc = evaluate_accuracy(model_k3, X_test, y_test)
    print("A6 Accuracy (k=3):", acc)

    # A7: Predict first 5 samples
    preds = predict_samples(model_k3, X_test)
    print("A7 Predictions (first 5):", preds)

    # A8: Accuracy vs k plot
    ks, accs = knn_accuracy_vs_k(X_train, X_test, y_train, y_test)
    plt.plot(ks, accs, marker='o')
    plt.title("Accuracy vs k")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.show()

    # A9: Confusion Matrix & Classification Report
    cm, report = evaluate_confusion_matrix(model_k3, X_test, y_test)
    print("A9 Confusion Matrix:\n", cm)
    print("Classification Report:\n", pd.DataFrame(report).T)
