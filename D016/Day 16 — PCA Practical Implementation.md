# Day 16 — PCA Practical Implementation

## 365-Day AIML Engineer / MNC-Oriented GitHub Challenge

### 📌 Topic

**Principal Component Analysis (PCA) — Practical Implementation**

---

## 🎯 Objective

The objective of Day 16 is to understand and implement **Principal Component Analysis (PCA)** for dimensionality reduction.

In this project, PCA is applied to the **Wine dataset** from Scikit-learn. The original dataset contains multiple features, which are standardized and then reduced to two principal components.

The reduced data is visualized and used with Logistic Regression to understand the practical impact of dimensionality reduction.

---

## 🧠 What is PCA?

**Principal Component Analysis (PCA)** is an unsupervised dimensionality-reduction technique used to transform a dataset with many correlated features into a smaller number of new features called **Principal Components**.

PCA tries to preserve as much information, or variance, from the original dataset as possible.

### Simple Example

Suppose a dataset contains:

```text
13 Original Features
        ↓
     PCA
        ↓
2 Principal Components
```

Instead of working with all 13 features, a machine learning model can work with only 2 transformed features while retaining a significant amount of the original information.

---

## 🔑 Why PCA is Used

PCA can be useful for:

* Reducing the number of features
* Reducing computational complexity
* Removing redundant information
* Handling highly correlated features
* Visualizing high-dimensional datasets
* Improving the efficiency of some machine learning pipelines
* Reducing storage requirements

---

## 📊 Dataset Used

This project uses the **Wine dataset** provided by Scikit-learn.

The dataset contains:

* **178 samples**
* **13 numerical features**
* **3 target classes**

Some of the features include:

* Alcohol
* Malic acid
* Ash
* Magnesium
* Total phenols
* Flavanoids
* Color intensity
* Proline

---

## 🛠️ Technologies Used

* Python
* NumPy
* Matplotlib
* Scikit-learn

---

## 📦 Libraries Used

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---

## 🔄 Project Workflow

The complete workflow is:

```text
Load Wine Dataset
       ↓
Separate Features and Target
       ↓
Train-Test Split
       ↓
Feature Standardization
       ↓
Apply PCA
       ↓
Reduce 13 Features → 2 Components
       ↓
Analyze Explained Variance
       ↓
Visualize PCA Data
       ↓
Train Logistic Regression
       ↓
Evaluate Model
       ↓
Compare Original Features vs PCA Features
```

---

# 1. Load the Dataset

The Wine dataset is loaded using Scikit-learn.

```python
from sklearn.datasets import load_wine

wine = load_wine()

X = wine.data
y = wine.target
```

Here:

* `X` contains the input features.
* `y` contains the target classes.

---

# 2. Train-Test Split

The dataset is divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The project uses:

* **80% training data**
* **20% testing data**

---

# 3. Feature Standardization

Before applying PCA, the features are standardized.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Why is standardization important?

Different features can have very different numerical ranges.

For example:

```text
Alcohol       → approximately 10–15
Magnesium     → approximately 70–160
Proline       → approximately 200–1700
```

Without scaling, features with larger numerical values could have an unnecessarily large influence on PCA.

---

# 4. Apply PCA

PCA is applied to reduce the original 13 features to 2 principal components.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
```

The transformation is:

```text
13 Original Features
          ↓
         PCA
          ↓
2 Principal Components
```

---

# 5. Explained Variance

PCA provides the amount of variance captured by each principal component.

```python
explained_variance = pca.explained_variance_ratio_
```

The explained variance ratio tells us how much information from the original dataset is represented by each component.

For example:

```text
PC1 → large percentage of variance
PC2 → additional percentage of variance
```

The total variance retained is calculated using:

```python
total_variance = np.sum(explained_variance)
```

---

# 6. Principal Components

PCA generates new features called **Principal Components**.

The components are combinations of the original features.

```python
pca.components_
```

For this project:

```text
Principal Component 1
Principal Component 2
```

are used as the two-dimensional representation of the original dataset.

---

# 7. PCA Visualization

The two principal components can be plotted using Matplotlib.

```python
plt.scatter(
    X_train_pca[y_train == class_value, 0],
    X_train_pca[y_train == class_value, 1]
)
```

The visualization allows us to see how the three wine classes are distributed after dimensionality reduction.

### Visualization Concept

```text
                 PC2
                  ↑
                  |
       Class 2    |       Class 1
          ● ●     |      ● ● ●
        ● ● ●     |    ● ● ●
                  |
------------------+----------------→ PC1
                  |
           ● ●    |
         ● ● ●    |   Class 3
```

The actual plot is generated automatically when the Python program runs.

---

# 8. Logistic Regression with PCA

After reducing the features, Logistic Regression is trained using the PCA-transformed data.

```python
model_pca = LogisticRegression(max_iter=1000)

model_pca.fit(X_train_pca, y_train)

y_pred_pca = model_pca.predict(X_test_pca)
```

---

# 9. Model Evaluation

The model is evaluated using accuracy and a classification report.

```python
accuracy_pca = accuracy_score(y_test, y_pred_pca)
```

The project also generates:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

# 10. Original Features vs PCA Features

The project also trains Logistic Regression using all original standardized features.

This allows us to compare:

| Approach         | Features |
| ---------------- | -------: |
| Original Dataset |       13 |
| PCA Dataset      |        2 |

The purpose of this comparison is to understand the trade-off between:

**Dimensionality reduction ↔ Information retention ↔ Model performance**

---

# 🧮 PCA Mathematical Concept

PCA works by finding directions in the data that capture maximum variance.

The first principal component captures the largest possible variance.

The second principal component captures the next largest variance while being orthogonal to the first.

Conceptually:

```text
Original Feature Space
        ↓
Find maximum variance direction
        ↓
Principal Component 1
        ↓
Find next maximum variance direction
        ↓
Principal Component 2
```

---

# 📁 Project Structure

```text
D16/
│
├── PCA_Practical_Implementation.py
│
└── Day 16 — PCA Practical Implementation.md
```

---

# ▶️ How to Run

Open Git Bash or Command Prompt inside the `D16` folder.

Run:

```bash
python PCA_Practical_Implementation.py
```

If `python` is not recognized, use:

```bash
py PCA_Practical_Implementation.py
```

---

# 📌 Expected Output

The program displays:

```text
PCA PRACTICAL IMPLEMENTATION

Dataset loaded successfully!

Number of samples: 178
Number of features: 13
Number of classes: 3
```

It then displays:

* Dataset information
* Standardized features
* PCA-transformed data
* Explained variance
* Principal component values
* Feature contributions
* Logistic Regression accuracy
* Classification report
* Confusion matrix
* Original vs PCA model comparison

A PCA scatter plot is also displayed.

---

# 💼 MNC / Industry Relevance

PCA is an important concept in practical machine learning because real-world datasets can contain hundreds or thousands of features.

PCA can help when:

* Datasets have many numerical features
* Features are highly correlated
* Visualization of high-dimensional data is required
* Computational efficiency is important
* Feature redundancy needs to be reduced

PCA is commonly discussed in areas such as:

* Machine Learning
* Data Science
* Computer Vision
* Pattern Recognition
* Signal Processing
* Recommendation Systems
* Exploratory Data Analysis

---

# ⚠️ Important Points to Remember

### 1. Standardize before PCA

For most datasets, feature scaling should be considered before PCA.

### 2. PCA creates new features

PCA does not simply select existing features.

It creates new features called principal components.

### 3. PCA is unsupervised

PCA does not use target labels while finding the principal components.

### 4. More components retain more information

Increasing the number of components generally increases the amount of variance retained.

### 5. PCA can reduce interpretability

The original feature names may become harder to interpret because principal components are combinations of multiple original features.

---

# 🧠 Key Learnings

After completing Day 16, I learned:

* What PCA is
* Why dimensionality reduction is useful
* How to standardize data
* How to apply PCA using Scikit-learn
* How to calculate explained variance
* How to interpret principal components
* How to visualize reduced-dimensional data
* How to train a classifier using PCA features
* How to compare PCA and non-PCA models
* How dimensionality reduction can support ML workflows

---

# 🚀 Day 16 Status

```text
Day 16 — PCA Practical Implementation

[✓] Dataset loaded
[✓] Data preprocessing
[✓] Feature standardization
[✓] PCA applied
[✓] Dimensionality reduced
[✓] Explained variance analyzed
[✓] Data visualized
[✓] Logistic Regression trained
[✓] Model evaluated
[✓] PCA vs original features compared
```

---

## 🔥 Next Step

**Day 16 completed — Principal Component Analysis (PCA) Practical Implementation.**

The next challenge will continue into the next distinct AIML topic without repeating PCA, K-Means, SVM, KNN, Decision Trees, or previous preprocessing tasks.
