# Day 23 — AdaBoost Classification

## 📌 Project Overview:

This project implements **AdaBoost (Adaptive Boosting)**, an ensemble machine learning algorithm used for classification tasks. AdaBoost combines multiple weak learners to build a stronger predictive model by giving more attention to incorrectly classified training samples.

In this project, the AdaBoost classifier is trained on the **Breast Cancer Wisconsin dataset** available in Scikit-learn.

## 🎯 Objectives

- Understand the fundamentals of ensemble learning.
- Learn how the AdaBoost algorithm works.
- Train a weak learner using a decision stump.
- Build an AdaBoost classification model.
- Evaluate model performance using classification metrics.
- Visualize the confusion matrix and feature importance.
- Make predictions on unseen test data.

## 🧠 What Is AdaBoost?

AdaBoost stands for **Adaptive Boosting**. It is an ensemble learning technique that combines several weak learners sequentially to create a stronger classifier.

A weak learner is a model that performs only slightly better than random guessing on a suitable classification task.

### How AdaBoost Works

1. Initialize weights for the training samples.
2. Train a weak learner on the weighted dataset.
3. Identify incorrectly classified samples.
4. Increase the importance of misclassified samples and reduce the relative importance of correctly classified samples.
5. Train the next weak learner using the updated sample weights.
6. Combine the learners' predictions using weighted voting.

The final prediction is determined by combining the contributions of the individual learners.

## 📚 Dataset Information

**Dataset:** Breast Cancer Wisconsin Dataset

**Source:** `sklearn.datasets.load_breast_cancer()`

The dataset contains measurements computed from digitized images of breast mass samples.

### Dataset Details

- **Total samples:** 569
- **Total features:** 30
- **Classes:** 2
- **Class labels:** Malignant and Benign
- **Training set:** 80%
- **Testing set:** 20%

The dataset is included in Scikit-learn, so a separate download is not required.

## 🛠️ Technologies Used

- Python
- NumPy
- Matplotlib
- Scikit-learn
- AdaBoost Classifier
- Decision Tree Classifier
- Classification Metrics

## 📂 Project Structure

```text
D23/
├── AdaBoost_Classification.py
└── Day 23 — AdaBoost Classification.md
```

## ⚙️ Implementation Steps

### 1. Import Libraries

Import NumPy, Matplotlib, and the required Scikit-learn classes and evaluation metrics.

### 2. Load the Dataset

Load the Breast Cancer Wisconsin dataset and separate the
