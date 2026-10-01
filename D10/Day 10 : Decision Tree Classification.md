# Day 10 — Decision Tree Classification

## 🎯 Objective

The objective of Day 10 was to learn and implement **Decision Tree Classification** using Python and Scikit-learn.

A Decision Tree is a supervised machine learning algorithm that makes predictions by splitting data into smaller groups using decision rules.

---

## 🧠 Topics Covered

* Decision Tree Classification
* Training and testing data
* `DecisionTreeClassifier`
* Gini impurity
* Maximum tree depth
* Model prediction
* Accuracy score
* Confusion matrix
* Classification report
* Feature importance
* Decision Tree visualization
* Prediction probability

---

## 📊 Dataset

For this task, I used the **Iris dataset** available through Scikit-learn.

The dataset contains measurements of iris flowers:

* Sepal length
* Sepal width
* Petal length
* Petal width

The model predicts the flower species:

* Setosa
* Versicolor
* Virginica

---

## ⚙️ Machine Learning Workflow

The workflow followed during Day 10 was:

```text
Load Dataset
     ↓
Separate Features and Target
     ↓
Train-Test Split
     ↓
Create Decision Tree Model
     ↓
Train Model
     ↓
Make Predictions
     ↓
Evaluate Model
     ↓
Visualize Decision Tree
     ↓
Test New Data
```

---

## 🌳 Decision Tree Model

The model was created using:

```python
DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)
```

### Parameters

**criterion = "gini"**

Gini impurity is used to measure how well the data is separated into different classes.

**max_depth = 3**

This limits the maximum depth of the tree and helps control overfitting.

**random_state = 42**

This makes the results reproducible
