# Day 11 - Feature Scaling & Standardization
# 365 Days AIML Engineer Challenge

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# --------------------------------------------------
# 1. Create Sample Dataset
# --------------------------------------------------

data = {
    "Age": [20, 25, 30, 35, 40],
    "Salary": [25000, 40000, 55000, 70000, 90000],
    "Experience": [1, 3, 5, 8, 12]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)


# --------------------------------------------------
# 2. Select Features
# --------------------------------------------------

features = ["Age", "Salary", "Experience"]

X = df[features]

print("\nOriginal Features:")
print(X)


# --------------------------------------------------
# 3. Standardization using StandardScaler
# --------------------------------------------------

standard_scaler = StandardScaler()

X_standardized = standard_scaler.fit_transform(X)

standardized_df = pd.DataFrame(
    X_standardized,
    columns=features
)

print("\nStandardized Data:")
print(standardized_df)


# --------------------------------------------------
# 4. Normalization using MinMaxScaler
# --------------------------------------------------

minmax_scaler = MinMaxScaler()

X_normalized = minmax_scaler.fit_transform(X)

normalized_df = pd.DataFrame(
    X_normalized,
    columns=features
)

print("\nNormalized Data:")
print(normalized_df)


# --------------------------------------------------
# 5. Compare Original and Scaled Data
# --------------------------------------------------

print("\nOriginal Data:")
print(X)

print("\nStandardized Data:")
print(standardized_df)

print("\nNormalized Data:")
print(normalized_df)


# --------------------------------------------------
# 6. Verify Standardization
# --------------------------------------------------

print("\nMean after Standardization:")
print(standardized_df.mean())

print("\nStandard Deviation after Standardization:")
print(standardized_df.std())


# --------------------------------------------------
# 7. Verify Normalization
# --------------------------------------------------

print("\nMinimum after Normalization:")
print(normalized_df.min())

print("\nMaximum after Normalization:")
print(normalized_df.max())


# --------------------------------------------------
# 8. Final Summary
# --------------------------------------------------

print("\n" + "=" * 50)
print("DAY 11 COMPLETE")
print("=" * 50)

print("""
Today I learned:

1. Feature Scaling
2. Standardization
3. Normalization
4. StandardScaler
5. MinMaxScaler
6. Why scaling is important in Machine Learning
""")