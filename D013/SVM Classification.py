# Day 13 - Support Vector Machine (SVM) Classification

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
# 2. Convert Dataset into DataFrame
# --------------------------------------------------

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

print("\nFirst 5 Rows:")
print(df.head())


# --------------------------------------------------
# 3. Split Dataset
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
# 5. Create SVM Model
# --------------------------------------------------

svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

svm_model.fit(X_train_scaled, y_train)


# --------------------------------------------------
# 6. Make Predictions
# --------------------------------------------------

y_pred = svm_model.predict(X_test_scaled)


# --------------------------------------------------
# 7. Calculate Accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n----------------------------------------")
print("SVM Classification Results")
print("----------------------------------------")

print("Kernel: RBF")
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
# 10. Compare Different SVM Kernels
# --------------------------------------------------

kernels = [
    "linear",
    "rbf",
    "poly"
]

kernel_accuracies = []

print("\n----------------------------------------")
print("SVM Kernel Comparison")
print("----------------------------------------")

for kernel in kernels:

    model = SVC(kernel=kernel)

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    score = accuracy_score(
        y_test,
        predictions
    )

    kernel_accuracies.append(score)

    print(
        f"Kernel = {kernel} --> "
        f"Accuracy = {score:.4f} "
        f"({score * 100:.2f}%)"
    )


# --------------------------------------------------
# 11. Create Results DataFrame
# --------------------------------------------------

results = pd.DataFrame({
    "Kernel": kernels,
    "Accuracy": kernel_accuracies
})

print("\nKernel Comparison:")
print(results)


# --------------------------------------------------
# 12. Find Highest Accuracy
# --------------------------------------------------

best_index = np.argmax(kernel_accuracies)

best_kernel = kernels[best_index]
best_accuracy = kernel_accuracies[best_index]

print("\n----------------------------------------")
print("Highest Accuracy Result")
print("----------------------------------------")

print("Kernel:", best_kernel)
print(
    "Accuracy:",
    round(best_accuracy * 100, 2),
    "%"
)


# --------------------------------------------------
# 13. Visualize Kernel Performance
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    kernels,
    kernel_accuracies
)

plt.title("SVM Kernel vs Accuracy")
plt.xlabel("SVM Kernel")
plt.ylabel("Accuracy")

plt.ylim(0, 1.1)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.show()