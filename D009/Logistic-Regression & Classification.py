# ============================================================
# Day 9: Logistic Regression Classification
# 365 Days AIML Engineer Challenge
# ============================================================

# 1. Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# 2. Load Dataset
# ============================================================

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["target"] = iris.target

print("\n========== DATASET ==========")
print(df.head())


# ============================================================
# 3. Explore Dataset
# ============================================================

print("\n========== DATASET INFORMATION ==========")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df["target"].value_counts())


# ============================================================
# 4. Convert into Binary Classification
# ============================================================

# Setosa = 1
# Not Setosa = 0

df["binary_target"] = (df["target"] == 0).astype(int)

print("\n========== BINARY TARGET ==========")
print(df[["target", "binary_target"]].head(10))

print("\nBinary Target Distribution:")
print(df["binary_target"].value_counts())


# ============================================================
# 5. Prepare Features and Target
# ============================================================

X = df[iris.feature_names]
y = df["binary_target"]

print("\n========== FEATURES AND TARGET ==========")
print("Features shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 6. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN TEST SPLIT ==========")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 7. Create Logistic Regression Model
# ============================================================

model = LogisticRegression(max_iter=200)


# ============================================================
# 8. Train Model
# ============================================================

model.fit(X_train, y_train)

print("\n========== MODEL TRAINING ==========")
print("Model training completed successfully!")


# ============================================================
# 9. Make Predictions
# ============================================================

y_pred = model.predict(X_test)

print("\n========== PREDICTIONS ==========")
print(y_pred)


# ============================================================
# 10. Compare Actual and Predicted Values
# ============================================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\n========== ACTUAL VS PREDICTED ==========")
print(results)


# ============================================================
# 11. Calculate Accuracy
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL ACCURACY ==========")
print("Accuracy:", accuracy)
print("Accuracy Percentage:", round(accuracy * 100, 2), "%")


# ============================================================
# 12. Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)


# ============================================================
# 13. Display Confusion Matrix
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Setosa", "Setosa"]
)

disp.plot()

plt.title("Logistic Regression - Confusion Matrix")
plt.show()


# ============================================================
# 14. Classification Report
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Setosa", "Setosa"]
    )
)


# ============================================================
# 15. Model Coefficients
# ============================================================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

print("\n========== MODEL COEFFICIENTS ==========")
print(coefficients)


# ============================================================
# 16. Prediction Probabilities
# ============================================================

probabilities = model.predict_proba(X_test)

probability_results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Probability_Not_Setosa": probabilities[:, 0],
    "Probability_Setosa": probabilities[:, 1]
})

print("\n========== PREDICTION PROBABILITIES ==========")
print(probability_results)


# ============================================================
# 17. Final Summary
# ============================================================

print("\n============================================")
print("           DAY 9 COMPLETED")
print("============================================")
print("Algorithm : Logistic Regression")
print("Problem   : Binary Classification")
print("Dataset   : Iris Dataset")
print("Accuracy  :", round(accuracy * 100, 2), "%")
print("============================================")
