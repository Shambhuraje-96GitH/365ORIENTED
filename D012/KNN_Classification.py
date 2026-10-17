
# ============================================================
# Day 12: K-Nearest Neighbors (KNN) Classification
# 365-Day AIML Engineer GitHub Challenge
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. Load the Iris Dataset
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target

print("=" * 60)
print("K-NEAREST NEIGHBORS (KNN) CLASSIFICATION")
print("=" * 60)

print("\nDataset loaded successfully!")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

print("\nClasses:")
for class_name in iris.target_names:
    print("-", class_name)


# ============================================================
# 2. Display Feature Names
# ============================================================

print("\nFeature Names:")

for feature in iris.feature_names:
    print("-", feature)


# ============================================================
# 3. Split Dataset into Training and Testing Data
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDataset Split:")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 4. Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed successfully!")


# ============================================================
# 5. Create and Train KNN Model
# ============================================================

k = 5

knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train_scaled, y_train)

print("\nKNN model trained successfully!")
print("K value:", k)


# ============================================================
# 6. Make Predictions
# ============================================================

y_pred = knn.predict(X_test_scaled)

print("\nPredictions:")
print(y_pred)


# ============================================================
# 7. Calculate Accuracy
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


# ============================================================
# 8. Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 9. Classification Report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# ============================================================
# 10. Test Different K Values
# ============================================================

k_values = [1, 3, 5, 7, 9]

accuracies = []

print("\n" + "=" * 60)
print("ACCURACY FOR DIFFERENT K VALUES")
print("=" * 60)

for k_value in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k_value
    )

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    score = accuracy_score(
        y_test,
        predictions
    )

    accuracies.append(score)

    print(
        f"K = {k_value} "
        f"-> Accuracy = {score * 100:.2f}%"
    )


# ============================================================
# 11. Find Best K Value
# ============================================================

best_index = np.argmax(accuracies)

best_k = k_values[best_index]
best_accuracy = accuracies[best_index]

print("\n" + "=" * 60)
print("BEST K VALUE")
print("=" * 60)

print("Best K value:", best_k)
print(f"Best Accuracy: {best_accuracy * 100:.2f}%")


# ============================================================
# 12. Visualize K Value vs Accuracy
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    accuracies,
    marker="o"
)

plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K Value vs Accuracy")