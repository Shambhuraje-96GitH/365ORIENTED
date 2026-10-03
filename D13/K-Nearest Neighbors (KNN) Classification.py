# Day 13 - K-Nearest Neighbors (KNN) Classification

# Import required libraries
import numpy as np
import pandas as pd
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


# --------------------------------------------------
# 1. Load the Iris Dataset
# --------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print("Feature Names:")
print(feature_names)

print("\nTarget Names:")
print(target_names)

print("\nDataset Shape:")
print(X.shape)


# --------------------------------------------------
# 2. Convert Dataset into a DataFrame
# --------------------------------------------------

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

print("\nFirst 5 Rows:")
print(df.head())


# --------------------------------------------------
# 3. Split Data into Training and Testing Sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# --------------------------------------------------
# 4. Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 5. Create KNN Model
# --------------------------------------------------

k = 5

knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train_scaled, y_train)


# --------------------------------------------------
# 6. Make Predictions
# --------------------------------------------------

y_pred = knn.predict(X_test_scaled)


# --------------------------------------------------
# 7. Calculate Accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n----------------------------------------")
print("KNN Classification Results")
print("----------------------------------------")

print("K Value:", k)
print("Accuracy:", accuracy)
print("Accuracy Percentage:", round(accuracy * 100, 2), "%")


# --------------------------------------------------
# 8. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# 9. Classification Report
# --------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_names
    )
)


# --------------------------------------------------
# 10. Test Different K Values
# --------------------------------------------------

k_values = [1, 3, 5, 7, 9]

accuracies = []

print("\n----------------------------------------")
print("Testing Different K Values")
print("----------------------------------------")

for k_value in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k_value
    )

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    score = accuracy_score(y_test, predictions)

    accuracies.append(score)

    print(
        f"K = {k_value} --> "
        f"Accuracy = {score:.4f} "
        f"({score * 100:.2f}%)"
    )


# --------------------------------------------------
# 11. Display K vs Accuracy
# --------------------------------------------------

results = pd.DataFrame({
    "K Value": k_values,
    "Accuracy": accuracies
})

print("\nK vs Accuracy:")
print(results)


# --------------------------------------------------
# 12. Find Highest Accuracy
# --------------------------------------------------

best_index = np.argmax(accuracies)

best_k = k_values[best_index]
best_accuracy = accuracies[best_index]

print("\n----------------------------------------")
print("Best Result")
print("----------------------------------------")

print("K Value:", best_k)
print("Accuracy:", round(best_accuracy * 100, 2), "%")


# --------------------------------------------------
# 13. Visualize K vs Accuracy
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    accuracies,
    marker="o"
)

plt.title("KNN: K Value vs Accuracy")
plt.xlabel("K Value")
plt.ylabel("Accuracy")

plt.xticks(k_values)
plt.grid(True)

plt.show()