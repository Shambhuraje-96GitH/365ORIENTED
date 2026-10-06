# ============================================================
# Day 16 — PCA Practical Implementation
# 365-Day AIML Engineer / MNC-Oriented GitHub Challenge
# ============================================================

# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 2. LOAD THE WINE DATASET
# ============================================================

wine = load_wine()

X = wine.data
y = wine.target

feature_names = wine.feature_names
target_names = wine.target_names

print("=" * 60)
print("PCA PRACTICAL IMPLEMENTATION")
print("=" * 60)

print("\nDataset loaded successfully!")

print("\nDataset Information:")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Number of classes:", len(target_names))

print("\nFeature Names:")
for feature in feature_names:
    print("-", feature)

print("\nTarget Classes:")
for target in target_names:
    print("-", target)


# ============================================================
# 3. DISPLAY ORIGINAL DATA
# ============================================================

print("\n" + "=" * 60)
print("ORIGINAL DATA")
print("=" * 60)

print("\nFirst 5 rows of features:")
print(X[:5])

print("\nFirst 5 target values:")
print(y[:5])


# ============================================================
# 4. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 5. STANDARDIZE THE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 60)
print("FEATURE STANDARDIZATION")
print("=" * 60)

print("\nFeatures standardized successfully.")

print("\nFirst 5 standardized training rows:")
print(X_train_scaled[:5])


# ============================================================
# 6. APPLY PCA
# ============================================================

# Reduce the original 13 features to 2 principal components.

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\n" + "=" * 60)
print("PCA DIMENSIONALITY REDUCTION")
print("=" * 60)

print("\nOriginal number of features:", X_train.shape[1])
print("Reduced number of features:", X_train_pca.shape[1])

print("\nFirst 5 PCA-transformed training rows:")
print(X_train_pca[:5])


# ============================================================
# 7. EXPLAINED VARIANCE
# ============================================================

explained_variance = pca.explained_variance_ratio_

print("\n" + "=" * 60)
print("EXPLAINED VARIANCE")
print("=" * 60)

print("\nExplained variance by Principal Components:")

for i, variance in enumerate(explained_variance, start=1):
    print(f"Principal Component {i}: {variance:.4f} "
          f"({variance * 100:.2f}%)")

total_variance = np.sum(explained_variance)

print("\nTotal variance retained by 2 components:",
      f"{total_variance * 100:.2f}%")


# ============================================================
# 8. PRINCIPAL COMPONENT INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("PRINCIPAL COMPONENTS")
print("=" * 60)

print("\nPrincipal Component 1:")
print(pca.components_[0])

print("\nPrincipal Component 2:")
print(pca.components_[1])


# ============================================================
# 9. DISPLAY FEATURE CONTRIBUTIONS
# ============================================================

print("\n" + "=" * 60)
print("FEATURE CONTRIBUTIONS TO PRINCIPAL COMPONENTS")
print("=" * 60)

print("\nFeature contributions:")

for i, feature in enumerate(feature_names):
    pc1_value = pca.components_[0][i]
    pc2_value = pca.components_[1][i]

    print(
        f"{feature:25s} "
        f"PC1: {pc1_value: .4f}   "
        f"PC2: {pc2_value: .4f}"
    )


# ============================================================
# 10. VISUALIZE PCA-TRANSFORMED DATA
# ============================================================

plt.figure(figsize=(10, 7))

for class_value in np.unique(y_train):

    plt.scatter(
        X_train_pca[y_train == class_value, 0],
        X_train_pca[y_train == class_value, 1],
        label=target_names[class_value],
        alpha=0.7
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("PCA — Wine Dataset")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 11. TRAIN LOGISTIC REGRESSION USING PCA FEATURES
# ============================================================

model_pca = LogisticRegression(max_iter=1000)

model_pca.fit(X_train_pca, y_train)

y_pred_pca = model_pca.predict(X_test_pca)


# ============================================================
# 12. EVALUATE PCA MODEL
# ============================================================

accuracy_pca = accuracy_score(y_test, y_pred_pca)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION WITH PCA")
print("=" * 60)

print("\nAccuracy using PCA features:",
      f"{accuracy_pca * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_pca,
        target_names=target_names
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_pca))


# ============================================================
# 13. TRAIN MODEL WITHOUT PCA FOR COMPARISON
# ============================================================

model_original = LogisticRegression(max_iter=1000)

model_original.fit(X_train_scaled, y_train)

y_pred_original = model_original.predict(X_test_scaled)

accuracy_original = accuracy_score(
    y_test,
    y_pred_original
)


# ============================================================
# 14. COMPARE ORIGINAL VS PCA MODEL
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print("\nOriginal number of features:",
      X_train_scaled.shape[1])

print("PCA number of features:",
      X_train_pca.shape[1])

print("\nAccuracy using all features:",
      f"{accuracy_original * 100:.2f}%")

print("Accuracy using PCA features:",
      f"{accuracy_pca * 100:.2f}%")


# ============================================================
# 15. EXPLAIN PCA BENEFIT
# ============================================================

print("\n" + "=" * 60)
print("PCA SUMMARY")
print("=" * 60)

print("\nPCA reduced the feature space from",
      X_train_scaled.shape[1],
      "features to",
      X_train_pca.shape[1],
      "features.")

print(
    f"\nThe first two principal components retained "
    f"{total_variance * 100:.2f}% of the total variance."
)

print("\nPCA can help machine learning systems by:")
print("1. Reducing dimensionality")
print("2. Removing redundant information")
print("3. Reducing computational complexity")
print("4. Helping visualize high-dimensional data")
print("5. Reducing the impact of correlated features")


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("DAY 16 PCA IMPLEMENTATION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nYou have successfully completed:")
print("✓ Wine dataset loading")
print("✓ Train-test splitting")
print("✓ Feature standardization")
print("✓ PCA dimensionality reduction")
print("✓ Explained variance analysis")
print("✓ Principal component analysis")
print("✓ PCA visualization")
print("✓ Logistic Regression with PCA")
print("✓ Model comparison")
print("✓ PCA performance evaluation")