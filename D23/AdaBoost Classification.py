
"""
Day 23: AdaBoost Classification
365-Day AIML Engineer GitHub Challenge

Description:
Train and evaluate an AdaBoost classifier using
the Breast Cancer Wisconsin dataset.
"""

import sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# 1. Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

print("Day 23: AdaBoost Classification")
print("-" * 45)
print("Dataset shape:", X.shape)
print("Classes:", data.target_names)

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 3. Create a weak learner
weak_learner = DecisionTreeClassifier(
    max_depth=1,
    random_state=42,
)

# 4. Build the AdaBoost model
# This API works with current scikit-learn versions.
model = AdaBoostClassifier(
    estimator=weak_learner,
    n_estimators=100,
    learning_rate=0.5,
    random_state=42,
)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation")
print("-" * 45)
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names,
    )
)

# 8. Display confusion matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names,
)

display.plot(cmap="Blues", values_format="d")
plt.title("Day 23 - AdaBoost Confusion Matrix")
plt.tight_layout()
plt.show()

# 9. Show feature importance
feature_importance = model.feature_importances_
indices = np.argsort(feature_importance)[-10:]

plt.figure(figsize=(10, 6))
plt.barh(
    data.feature_names[indices],
    feature_importance[indices],
)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 10 Features - AdaBoost")
plt.tight_layout()
plt.show()

# 10. Predict a sample
sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)[0]

print("\nSample Prediction")
print("-" * 45)
print("Predicted class:", data.target_names[prediction])
print("Actual class:", data.target_names[y_test[0]])

print("\nDay 23 completed successfully!")
