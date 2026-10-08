# Day 17 — PCA Visualization & Explained Variance Analysis

## 365-Day AIML / MNC-Oriented GitHub Challenge

---

## 📌 Project Title

**PCA Visualization & Explained Variance Analysis**

---

## 🎯 Objective

The objective of Day 17 is to understand how **Principal Component Analysis (PCA)** can be used to analyze high-dimensional datasets and visualize them in a lower-dimensional space.

In this practical implementation, the **Wine dataset** from Scikit-learn is used to:

* Load a real-world machine learning dataset
* Explore the dataset
* Standardize numerical features
* Apply PCA
* Calculate explained variance
* Calculate cumulative explained variance
* Determine the number of components required to retain 90% and 95% of the variance
* Reduce the dataset to two principal components
* Visualize the transformed data in 2D
* Save the PCA-transformed dataset as a CSV file

---

## 🧠 What is PCA?

**Principal Component Analysis (PCA)** is an unsupervised dimensionality reduction technique.

It transforms a dataset containing many correlated features into a smaller set of new variables called **Principal Components**.

Each principal component captures a portion of the information or variance present in the original dataset.

### Example

Suppose a dataset contains:

```text
13 original features
        ↓
       PCA
        ↓
2 principal components
```

The dataset can then be visualized in a 2D graph while retaining as much information as possible.

---

## 🔑 Why PCA is Important in AIML

PCA is widely used in machine learning and data science for:

* Dimensionality reduction
* Data visualization
* Feature compression
* Noise reduction
* Removing feature correlation
* Improving computational efficiency
* Reducing storage requirements
* Preparing high-dimensional data for machine learning models

PCA is especially useful when a dataset contains many numerical features.

---

# 📊 Dataset Used

The project uses the **Wine dataset** provided by Scikit-learn.

The dataset contains chemical measurements of wines belonging to three different classes.

### Dataset characteristics

| Property                              | Value |
| ------------------------------------- | ----: |
| Samples                               |   178 |
| Original Features                     |    13 |
| Classes                               |     3 |
| PCA Components Used for Visualization |     2 |

The dataset contains features such as:

* Alcohol
* Malic acid
* Ash
* Alcalinity of ash
* Magnesium
* Total phenols
* Flavanoids
* Nonflavanoid phenols
* Proanthocyanins
* Color intensity
* Hue
* OD280/OD315 of diluted wines
* Proline

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

# 📦 Required Libraries

Install the required packages using:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

# 📁 Project Structure

```text
D17/
│
├── PCA_Visualization.py
│
├── Day 17 — PCA Visualization & Explained Variance Analysis.md
│
└── PCA_2D_Transformed_Data.csv
```

The CSV file is automatically generated when the Python program is executed.

---

# 🔄 Project Workflow

The complete workflow is:

```text
Load Wine Dataset
        ↓
Create Pandas DataFrame
        ↓
Explore Dataset
        ↓
Separate Features and Target
        ↓
Standardize Features
        ↓
Apply PCA
        ↓
Calculate Explained Variance
        ↓
Calculate Cumulative Variance
        ↓
Find Components for 90% Variance
        ↓
Find Components for 95% Variance
        ↓
Reduce Dataset to 2 Components
        ↓
Visualize PCA Data
        ↓
Save Transformed Dataset
```

---

# 1️⃣ Import Required Libraries

The project imports:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

### Purpose

| Library      | Purpose                  |
| ------------ | ------------------------ |
| NumPy        | Numerical operations     |
| Pandas       | Data manipulation        |
| Matplotlib   | Data visualization       |
| Scikit-learn | Dataset, scaling and PCA |

---

# 2️⃣ Load the Wine Dataset

The Wine dataset is loaded using:

```python
wine = load_wine()

X = wine.data
y = wine.target
```

The dataset contains:

```text
178 samples
13 features
3 classes
```

---

# 3️⃣ Create a DataFrame

The features are converted into a Pandas DataFrame:

```python
df = pd.DataFrame(X, columns=feature_names)

df["target"] = y
```

This makes the dataset easier to inspect and analyze.

---

# 4️⃣ Explore the Dataset

The program displays:

```python
df.head()
```

and:

```python
df.describe()
```

This provides information about:

* Feature values
* Mean
* Standard deviation
* Minimum
* Maximum
* Quartiles

---

# 5️⃣ Separate Features and Target

The target column is separated from the input features:

```python
X = df.drop("target", axis=1)
y = df["target"]
```

Here:

```text
X → Input features
y → Target classes
```

---

# 6️⃣ Feature Standardization

Before applying PCA, the features are standardized.

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Standardization transforms the features so that they have approximately:

```text
Mean = 0
Standard Deviation = 1
```

### Why is standardization important?

PCA is affected by feature scale.

For example:

```text
Feature A → values between 0 and 1

Feature B → values between 0 and 10,000
```

Without standardization, Feature B could have a much larger influence on PCA.

Therefore:

```text
Original Data
     ↓
StandardScaler
     ↓
Standardized Data
     ↓
PCA
```

---

# 7️⃣ Apply PCA

PCA is initially applied without specifying the number of components:

```python
pca_full = PCA()

X_pca_full = pca_full.fit_transform(X_scaled)
```

This allows us to analyze all principal components.

---

# 8️⃣ Explained Variance Ratio

The explained variance ratio tells us how much information each principal component captures.

```python
explained_variance = pca_full.explained_variance_ratio_
```

For example:

```text
PC1 → 36.20%
PC2 → 19.20%
PC3 → 11.10%
...
```

The actual values are calculated by the program.

### Interpretation

If:

```text
PC1 = 36%
```

it means the first principal component captures approximately 36% of the total variance in the dataset.

---

# 9️⃣ Cumulative Explained Variance

The cumulative variance is calculated using:

```python
cumulative_variance = np.cumsum(explained_variance)
```

This tells us how much total information is retained when multiple components are combined.

Example:

```text
PC1              → 36%
PC1 + PC2        → 55%
PC1 + PC2 + PC3  → 66%
```

The program prints the cumulative explained variance for every component.

---

# 🔟 Selecting Components for 90% Variance

The program determines how many components are required to retain at least 90% of the original variance:

```python
components_90 = np.argmax(
    cumulative_variance >= 0.90
) + 1
```

This is useful when deciding how many dimensions should be retained after PCA.

---

# 1️⃣1️⃣ Selecting Components for 95% Variance

The same process is performed for 95% variance:

```python
components_95 = np.argmax(
    cumulative_variance >= 0.95
) + 1
```

This provides another practical dimensionality-reduction threshold.

---

# 📈 12. Cumulative Explained Variance Visualization

The program creates a graph showing:

```text
Number of Principal Components
              vs
Cumulative Explained Variance
```

The graph also contains reference lines for:

```text
90% variance
95% variance
```

This visualization helps identify how many components are needed to preserve most of the information.

### Graph interpretation

The closer the curve gets to:

```text
1.0 = 100% variance
```

the more information is retained.

---

# 1️⃣3️⃣ PCA with Two Components

For visualization, PCA is applied using two components:

```python
pca_2d = PCA(n_components=2)

X_pca_2d = pca_2d.fit_transform(X_scaled)
```

The original dataset has:

```text
13 features
```

After PCA:

```text
2 principal components
```

Therefore:

```text
13D
 ↓
PCA
 ↓
2D
```

---

# 📊 14. Explained Variance of Two Components

The program calculates the variance explained by PC1 and PC2:

```python
pca_2d.explained_variance_ratio_
```

It also calculates their combined variance:

```python
pca_2d.explained_variance_ratio_.sum()
```

This tells us how much of the original information is represented in the 2D visualization.

---

# 1️⃣5️⃣ Create PCA DataFrame

The two principal components are stored in a new DataFrame:

```python
pca_df = pd.DataFrame(
    X_pca_2d,
    columns=["PC1", "PC2"]
)
```

The target class is then added:

```python
pca_df["target"] = y.values
```

The resulting structure is:

```text
PC1 | PC2 | target
----|-----|-------
... | ... | 0
... | ... | 1
... | ... | 2
```

---

# 📍 16. 2D PCA Visualization

The transformed dataset is visualized using a scatter plot.

The three Wine classes are plotted separately.

```python
plt.scatter(
    points["PC1"],
    points["PC2"]
)
```

The resulting graph allows us to visually inspect whether different classes are separated in PCA space.

### Interpretation

If different classes form separate groups, PCA has successfully created a lower-dimensional representation that preserves useful class-related structure.

---

# 📉 17. Dimensionality Reduction

The program compares:

```text
Original features = 13
Reduced features  = 2
```

The percentage reduction is calculated using:

```python
reduction_percentage = (
    (original_features - reduced_features)
    / original_features
) * 100
```

This demonstrates the practical benefit of dimensionality reduction.

---

# 💾 18. Save PCA Dataset

The transformed 2D dataset is saved using:

```python
pca_df.to_csv(
    "PCA_2D_Transformed_Data.csv",
    index=False
)
```

The generated file is:

```text
PCA_2D_Transformed_Data.csv
```

This file can be used for further analysis or machine learning experiments.

---

# ▶️ How to Run

Open the terminal inside the `D17` directory.

Run:

```bash
python PCA_Visualization.py
```

The program will:

1. Load the Wine dataset
2. Display dataset information
3. Standardize the features
4. Perform PCA
5. Calculate explained variance
6. Calculate cumulative variance
7. Determine 90% and 95% component requirements
8. Display the cumulative variance graph
9. Perform 2D PCA
10. Display the PCA scatter plot
11. Save the transformed data as CSV

---

# 📌 Expected Output

The terminal will display information similar to:

```text
DAY 17 — PCA VISUALIZATION & EXPLAINED VARIANCE ANALYSIS

Dataset loaded successfully!

Dataset Information:
Number of samples : 178
Number of features: 13
Number of classes : 3

STANDARDIZATION

Feature standardization completed.

PCA ANALYSIS

Explained Variance Ratio:
Principal Component 1: ...
Principal Component 2: ...
Principal Component 3: ...
...
```

It will also display:

```text
Components required to retain at least 90% variance: ...

Components required to retain at least 95% variance: ...
```

And:

```text
DIMENSIONALITY REDUCTION SUMMARY

Original number of features : 13
Reduced number of features  : 2
Dimensionality reduction    : ...
```

---

# 📊 Visualizations Generated

The program generates two visualizations.

## Visualization 1 — Cumulative Explained Variance

Shows how much variance is retained as more principal components are added.

```text
Components → Explained Variance
```

This helps determine an appropriate number of PCA components.

---

## Visualization 2 — 2D PCA Scatter Plot

Shows the Wine dataset after reducing the original 13 dimensions to:

```text
PC1
PC2
```

The three wine classes are displayed separately.

---

# 🧠 Key Concepts Learned

### 1. Dimensionality Reduction

Reducing the number of features while preserving important information.

### 2. Standardization

Putting features on a comparable scale before PCA.

### 3. Principal Components

New dimensions created by PCA that capture maximum variance.

### 4. Explained Variance

Measures how much information each principal component retains.

### 5. Cumulative Explained Variance

Measures the total variance retained by multiple components.

### 6. PCA Visualization

Allows high-dimensional data to be visualized in two or three dimensions.

---

# 💼 MNC / AIML Interview Relevance

PCA is an important concept for AIML and Data Science interviews.

### Common Interview Questions

**Q1. What is PCA?**

PCA is a dimensionality-reduction technique that transforms correlated features into a smaller number of uncorrelated principal components.

---

**Q2. Why do we standardize data before PCA?**

Because PCA is sensitive to feature scale. Standardization prevents features with larger numerical ranges from dominating the analysis.

---

**Q3. What is explained variance?**

Explained variance represents the proportion of total dataset variance captured by each principal component.

---

**Q4. What is cumulative explained variance?**

It is the total variance retained when multiple principal components are considered together.

---

**Q5. How do you decide the number of PCA components?**

A common approach is to choose enough components to retain a desired percentage of variance, such as 90%, 95%, or 99%.

---

**Q6. Is PCA supervised or unsupervised?**

PCA is an **unsupervised dimensionality-reduction technique** because it does not use target labels while finding the principal components.

---

**Q7. Does PCA always improve model performance?**

No. PCA can reduce dimensionality and computational cost, but it can also remove information that may be useful for a particular model.

---

# ⚠️ Important Considerations

PCA should generally be applied carefully.

### Standardize features

Always consider feature scaling when features have different units or ranges.

### PCA reduces interpretability

The original features are transformed into combinations called principal components, which can make interpretation more difficult.

### Information can be lost

Reducing dimensions means some variance may be discarded.

### PCA works primarily with numerical data

Categorical variables require appropriate preprocessing before PCA.

---

# 🔬 Practical AIML Pipeline

The Day 17 implementation demonstrates the following real-world pipeline:

```text
Raw Dataset
     ↓
Data Exploration
     ↓
Feature / Target Separation
     ↓
Feature Scaling
     ↓
PCA
     ↓
Explained Variance Analysis
     ↓
Component Selection
     ↓
Dimensionality Reduction
     ↓
Visualization
     ↓
Export Transformed Data
```

---

# 📈 Before vs After PCA

| Property            |             Before PCA |                           After PCA |
| ------------------- | ---------------------: | ----------------------------------: |
| Number of Features  |                     13 |                                   2 |
| Representation      |      Original Features |                Principal Components |
| Visualization       |              Difficult |               Easy 2D Visualization |
| Feature Correlation |                Present | Principal components are orthogonal |
| Information         | 100% original variance |      Depends on selected components |

---

# 🏆 Day 17 Learning Outcome

After completing this project, you should understand:

* What PCA is
* Why PCA is used
* Why standardization is important
* How PCA transforms features
* How explained variance works
* How cumulative variance is calculated
* How to select PCA components
* How to visualize high-dimensional data
* How to reduce 13 features to 2 dimensions
* How to export PCA-transformed data

---

# 🚀 Future Applications

The concepts learned today can be applied to:

* Image compression
* Face recognition
* Computer vision
* Customer segmentation
* Financial data analysis
* Genomics
* NLP feature reduction
* Anomaly detection
* Machine learning preprocessing
* High-dimensional data visualization

---

# 📌 GitHub Commit

After testing the Python file and confirming that the graphs and CSV are generated successfully:

```bash
git add D17/
```

Then commit:

```bash
git commit -m "Day 17: PCA Visualization and Explained Variance Analysis"
```

Finally push:

```bash
git push origin main
```

If GitHub rejects the push because the remote contains newer commits:

```bash
git pull --rebase origin main
```

Then:

```bash
git push origin main
```

---

# ✅ Day 17 Completed

**Topic:** PCA Visualization & Explained Variance Analysis

**Dataset:** Wine Dataset

**Original Dimensions:** 13 features

**Visualization Dimensions:** 2 principal components

**Main Concepts:**

```text
Standardization
      ↓
PCA
      ↓
Explained Variance
      ↓
Cumulative Variance
      ↓
Component Selection
      ↓
2D Visualization
      ↓
Dimensionality Reduction
```

---

## 🎯 Challenge Progress

```text
Day 1  → NumPy Basics
Day 2  → Pandas Basics
Day 3  → NumPy Array Operations
...
Day 13 → SVM Classification
Day 14 → K-Means Clustering
Day 15 → PCA
Day 16 → PCA Practical Implementation
Day 17 → PCA Visualization & Explained Variance Analysis
```

**365-Day AIML / MNC Engineer Challenge — Day 17 Complete.**
