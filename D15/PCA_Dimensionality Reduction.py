# ============================================================
# Day 15 - PCA (Principal Component Analysis)
# AIML Engineer - 365 Days GitHub Challenge
# ============================================================

# PCA is a dimensionality reduction technique.
# It converts a dataset with many features into fewer
# meaningful components while preserving as much information
# as possible.


# ------------------------------------------------------------
# 1. Import Required Libraries
# ------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ------------------------------------------------------------
# 2. Load the Iris Dataset
# ------------------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print("Original Feature Names:")
print(feature_names)

print("\nDataset Shape:")
print(X.shape)


# ------------------------------------------------------------
# 3. Convert Dataset into a DataFrame
# ------------------------------------------------------------

df = pd.DataFrame(X, columns=feature_names)

df["target"] = y

print("\nFirst 5 Rows:")
print(df.head())


# ------------------------------------------------------------
# 4. Separate Features and Target
# ------------------------------------------------------------

X = df[feature_names]
y = df["target"]


# ------------------------------------------------------------
# 5. Standardize the Features
# ------------------------------------------------------------

# PCA is affected by the scale of features.
# Therefore, standardization is performed before PCA.

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nStandardized Data:")
print(X_scaled[:5])


# ------------------------------------------------------------
# 6. Apply PCA
# ------------------------------------------------------------

# Original dataset has 4 features.
# We reduce them to 2 principal components.

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)


# ------------------------------------------------------------
# 7. Create PCA DataFrame
# ------------------------------------------------------------

pca_df = pd.DataFrame(
    X_pca,
    columns=["Principal Component 1", "Principal Component 2"]
)

pca_df["target"] = y

print("\nPCA Transformed Data:")
print(pca_df.head())


# ------------------------------------------------------------
# 8. Check Explained Variance
# ------------------------------------------------------------

explained_variance = pca.explained_variance_ratio_

print("\nExplained Variance Ratio:")
print(explained_variance)

print("\nTotal Explained Variance:")
print(explained_variance.sum())


# ------------------------------------------------------------
# 9. Display PCA Components
# ------------------------------------------------------------

print("\nPCA Components:")
print(pca.components_)


# ------------------------------------------------------------
# 10. Visualize PCA Results
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

for target in range(3):

    plt.scatter(
        pca_df.loc[pca_df["target"] == target, "Principal Component 1"],
        pca_df.loc[pca_df["target"] == target, "Principal Component 2"],
        label=target_names[target]
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("Iris Dataset After PCA")

plt.legend()

plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 11. Compare Original and Reduced Dimensions
# ------------------------------------------------------------

print("\nOriginal Number of Features:")
print(X.shape[1])

print("\nReduced Number of Features:")
print(X_pca.shape[1])


# ------------------------------------------------------------
# 12. Final Summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DAY 15 - PCA COMPLETED")
print("=" * 60)

print("Original dimensions :", X.shape[1])
print("Reduced dimensions  :", X_pca.shape[1])

print(
    "Information retained :",
    round(explained_variance.sum() * 100, 2),
    "%"
)

print("\nPCA successfully reduced the Iris dataset")
print("from 4 features to 2 principal components.")