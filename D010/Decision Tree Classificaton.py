# ============================================
# DAY 10 - DECISION TREE CLASSIFICATION
# 365 Days GitHub AIML Challenge
# ============================================

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================
# 1. Load Dataset
# ============================================

iris = load_iris()

X = iris.data
y = iris.target

print("Feature names:")
print(iris.feature_names)

print("\nTarget names:")
print(iris.target_names)

print("\nDataset shape:")
print(X.shape)


# ============================================
# 2. Display Sample Data
# ============================================

print("\nFirst 5 rows of X:")
print(X[:5])

print("\nFirst 5 target values:")
print(y[:5])


# ============================================
# 3. Split Dataset
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# 4. Create Decision Tree Model
# ============================================

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)


# ============================================
# 5. Train Model
# ============================================

model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")


# ============================================
# 6. Make Predictions
# ============================================

y_pred = model.predict(X_test)

print("\nPredicted values:")
print(y_pred)

print("\nActual values:")
print(y_test)


# ============================================
# 7. Calculate Accuracy
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)


# ============================================
# 8. Confusion Matrix
# ============================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================
# 9. Classification Report
# ============================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# ============================================
# 10. Feature Importance
# ============================================

print("\nFeature Importance:")

for feature, importance in zip(
    iris.feature_names,
    model.feature_importances_
):
    print(f"{feature}: {importance:.4f}")


# ============================================
# 11. Visualize Decision Tree
# ============================================

plt.figure(figsize=(15, 8))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

plt.title("Decision Tree Classifier")

plt.show()


# ============================================
# 12. Test a New Flower
# ============================================

new_flower = np.array([
    [5.1, 3.5, 1.4, 0.2]
])

prediction = model.predict(new_flower)

print("\nNew Flower Prediction:")
print(iris.target_names[prediction[0]])


# ============================================
# 13. Prediction Probability
# ============================================

probability = model.predict_proba(new_flower)

print("\nPrediction Probability:")
print(probability)


# ============================================
# END OF DAY 10
# ============================================