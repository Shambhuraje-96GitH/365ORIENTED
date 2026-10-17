# Day 15 — PCA (Principal Component Analysis)

## 📌 Overview

Today I learned **Principal Component Analysis (PCA)**, a dimensionality reduction technique used in Machine Learning.

PCA helps reduce the number of features in a dataset while preserving as much important information as possible.

In this task, I applied PCA to the **Iris dataset** and reduced its original 4 features to 2 principal components.

---

## 🎯 Objectives

* Understand Principal Component Analysis (PCA)
* Understand dimensionality reduction
* Standardize features before applying PCA
* Apply PCA using Scikit-learn
* Reduce 4 features to 2 principal components
* Calculate explained variance
* Visualize the PCA-transformed dataset

---

## 🛠️ Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn

---

## 📂 Files

```text
D15/
│
├── PCA_Dimensionality_Reduction.py
└── README.md
```

---

## 📊 Dataset

The **Iris dataset** from Scikit-learn was used.

The original dataset contains 4 features:

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

The dataset contains 150 samples belonging to 3 different Iris flower species.

---

## 🔄 Workflow

```text
Load Iris Dataset
        ↓
Separate Features and Target
        ↓
Standardize Features
        ↓
Apply PCA
        ↓
Reduce 4 Features → 2 Components
        ↓
Calculate Explained Variance
        ↓
Visualize PCA Results
```

---

## 🧠 What I Learned

### 1. Dimensionality Reduction

Dimensionality reduction means reducing the number of features in a dataset while trying to preserve the important information.

It can help with:

* Faster model training
* Data visualization
* Removing redundant information
* Handling high-dimensional datasets

### 2. Standardization

Before applying PCA, the features were standardized using `StandardScaler`.

This is important because PCA is affected by the scale of the features.

### 3. Principal Components

PCA transforms the original features into new features called **principal components**.

The first principal component captures the greatest amount of variance, while the second captures the next greatest amount.

### 4. Explained Variance

The explained variance ratio tells us how much information is retained by each principal component.

In this project, the first two components were selected for visualization.

---

## 📈 Visualization

The PCA-transformed Iris dataset was plotted using:

* Principal Component 1 on the X-axis
* Principal Component 2 on the Y-axis

The three Iris species were represented as separate groups.

---

## ▶️ How to Run

Open the terminal inside the `D15` folder and run:

```bash
python PCA_Dimensionality_Reduction.py
```

---

## 💡 Key Takeaway

> PCA is a powerful dimensionality reduction technique that transforms many original features into a smaller number of principal components while retaining important information.

---

## 🚀 AIML Engineer Relevance

PCA is useful in real-world Machine Learning and AI projects for:

* Feature engineering
* High-dimensional datasets
* Data visualization
* Noise reduction
* Improving computational efficiency
* Preparing data for Machine Learning models

---

## 📅 365-Day GitHub Challenge

**Day:** 15
**Topic:** PCA — Principal Component Analysis
**Track:** AIML Engineer / MNC-Oriented Machine Learning

---

## ✅ Status

**Completed — Day 15 🎯**
