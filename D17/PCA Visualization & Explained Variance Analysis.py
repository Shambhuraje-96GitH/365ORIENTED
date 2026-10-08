"""
Day 17 — PCA Visualization & Explained Variance Analysis

365-Day AIML / MNC-Oriented GitHub Challenge

Objective:
    - Load the Wine dataset
    - Standardize the features
    - Apply PCA
    - Analyze explained variance
    - Determine the number of useful components
    - Visualize cumulative explained variance
    - Visualize the dataset in 2D PCA space

Libraries:
    numpy
    pandas
    matplotlib
    scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("DAY 17 — PCA VISUALIZATION & EXPLAINED VARIANCE ANALYSIS")
print("=" * 60)

wine = load_wine()

X = wine.data
y = wine.target

feature_names = wine.feature_names
target_names = wine.target_names

print("\nDataset loaded successfully!")

print("\nDataset Information:")
print(f"Number of samples : {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes : {len(target_names)}")


# ============================================================
# 2. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(X, columns=feature_names)

df["target"] = y

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. CHECK BASIC STATISTICS
# ============================================================

print("\nBasic Statistics:")
print(df.describe())


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("target", axis=1)
y = df["target"]


# ============================================================
# 5. STANDARDIZE FEATURES
# ============================================================

print("\n" + "-" * 60)
print("STANDARDIZATION")
print("-" * 60)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Feature standardization completed.")

print("\nMean of standardized features:")
print(np.round(X_scaled.mean(axis=0), 4))

print("\nStandard deviation of standardized features:")
print(np.round(X_scaled.std(axis=0), 4))


# ============================================================
# 6. APPLY PCA WITH ALL COMPONENTS
# ============================================================

print("\n" + "-" * 60)
print("PCA ANALYSIS")
print("-" * 60)

pca_full = PCA()

X_pca_full = pca_full.fit_transform(X_scaled)

explained_variance = pca_full.explained_variance_ratio_

print("\nExplained Variance Ratio:")

for i, variance in enumerate(explained_variance, start=1):
    print(f"Principal Component {i}: {variance:.4f} "
          f"({variance * 100:.2f}%)")


# ============================================================
# 7. CUMULATIVE EXPLAINED VARIANCE
# ============================================================

cumulative_variance = np.cumsum(explained_variance)

print("\nCumulative Explained Variance:")

for i, variance in enumerate(cumulative_variance, start=1):
    print(f"First {i} component(s): {variance * 100:.2f}%")


# ============================================================
# 8. FIND COMPONENTS REQUIRED FOR 90% VARIANCE
# ============================================================

components_90 = np.argmax(cumulative_variance >= 0.90) + 1

print("\n" + "-" * 60)
print("COMPONENT SELECTION")
print("-" * 60)

print(
    f"Components required to retain at least 90% variance: "
    f"{components_90}"
)

print(
    f"Variance retained: "
    f"{cumulative_variance[components_90 - 1] * 100:.2f}%"
)


# ============================================================
# 9. FIND COMPONENTS REQUIRED FOR 95% VARIANCE
# ============================================================

components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(
    f"\nComponents required to retain at least 95% variance: "
    f"{components_95}"
)

print(
    f"Variance retained: "
    f"{cumulative_variance[components_95 - 1] * 100:.2f}%"
)


# ============================================================
# 10. VISUALIZATION — EXPLAINED VARIANCE
# ============================================================

plt.figure(figsize=(10, 6))

components = range(1, len(explained_variance) + 1)

plt.plot(
    components,
    cumulative_variance,
    marker="o",
    linewidth=2
)

plt.axhline(
    y=0.90,
    linestyle="--",
    label="90% Variance"
)

plt.axhline(
    y=0.95,
    linestyle="--",
    label="95% Variance"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")

plt.title("PCA — Cumulative Explained Variance")

plt.xticks(components)
plt.ylim(0, 1.05)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 11. PCA WITH 2 COMPONENTS
# ============================================================

print("\n" + "-" * 60)
print("2D PCA TRANSFORMATION")
print("-" * 60)

pca_2d = PCA(n_components=2)

X_pca_2d = pca_2d.fit_transform(X_scaled)

print("Original feature dimensions:", X_scaled.shape)
print("Reduced feature dimensions:", X_pca_2d.shape)

print("\nVariance explained by PC1:")
print(f"{pca_2d.explained_variance_ratio_[0] * 100:.2f}%")

print("\nVariance explained by PC2:")
print(f"{pca_2d.explained_variance_ratio_[1] * 100:.2f}%")

print("\nTotal variance explained by 2 components:")
print(
    f"{pca_2d.explained_variance_ratio_.sum() * 100:.2f}%"
)


# ============================================================
# 12. CREATE PCA DATAFRAME
# ============================================================

pca_df = pd.DataFrame(
    X_pca_2d,
    columns=["PC1", "PC2"]
)

pca_df["target"] = y.values

print("\nPCA-transformed data:")
print(pca_df.head())


# ============================================================
# 13. VISUALIZE DATA IN 2D PCA SPACE
# ============================================================

plt.figure(figsize=(10, 7))

for target_value, target_name in enumerate(target_names):

    points = pca_df[pca_df["target"] == target_value]

    plt.scatter(
        points["PC1"],
        points["PC2"],
        label=target_name,
        alpha=0.7,
        s=60
    )


plt.xlabel(
    f"Principal Component 1 "
    f"({pca_2d.explained_variance_ratio_[0] * 100:.2f}% variance)"
)

plt.ylabel(
    f"Principal Component 2 "
    f"({pca_2d.explained_variance_ratio_[1] * 100:.2f}% variance)"
)

plt.title("Wine Dataset — 2D PCA Visualization")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()


# ============================================================
# 14. COMPARE ORIGINAL VS REDUCED DIMENSIONS
# ============================================================

original_features = X_scaled.shape[1]
reduced_features = X_pca_2d.shape[1]

reduction_percentage = (
    (original_features - reduced_features)
    / original_features
) * 100

print("\n" + "-" * 60)
print("DIMENSIONALITY REDUCTION SUMMARY")
print("-" * 60)

print(f"Original number of features : {original_features}")
print(f"Reduced number of features  : {reduced_features}")
print(f"Dimensionality reduction    : {reduction_percentage:.2f}%")

print(
    f"\nInformation retained by 2 components: "
    f"{pca_2d.explained_variance_ratio_.sum() * 100:.2f}%"
)


# ============================================================
# 15. SAVE PCA DATASET
# ============================================================

pca_df.to_csv(
    "PCA_2D_Transformed_Data.csv",
    index=False
)

print("\nPCA-transformed dataset saved as:")
print("PCA_2D_Transformed_Data.csv")


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DAY 17 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nKey Results:")

print(f"Original dimensions : {original_features}")
print(f"2D dimensions       : {reduced_features}")

print(
    f"2D variance retained: "
    f"{pca_2d.explained_variance_ratio_.sum() * 100:.2f}%"
)

print(
    f"Components for 90% variance: "
    f"{components_90}"
)

print(
    f"Components for 95% variance: "
    f"{components_95}"
)

print("\nPCA visualization completed.")
print("CSV output generated successfully.")