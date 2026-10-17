# Day 12 — K-Nearest Neighbors (KNN) Classification

## Objective

The objective of Day 12 is to understand and implement the **K-Nearest Neighbors (KNN)** classification algorithm using Python and Scikit-learn.

KNN is a supervised machine learning algorithm commonly used for classification problems.

---

## What is KNN?

**K-Nearest Neighbors (KNN)** is a machine learning algorithm that classifies a new data point based on the classes of its nearest neighboring data points.

The main idea is:

> Data points that are close to each other are likely to belong to the same class.

For classification, KNN finds the nearest data points and uses majority voting to determine the predicted class.

---

## How KNN Works

The basic process is:

```text
New Data Point
      ↓
Calculate Distance
      ↓
Find K Nearest Neighbors
      ↓
Check Their Classes
      ↓
Majority Voting
      ↓
Predicted Class
```

---

## What is K?

`K` represents the number of nearest neighbors considered during prediction.

For example:

```text
K = 5
```

means that the algorithm considers the five closest training data points.

The class occurring most frequently among those neighbors becomes the predicted class.

---

## Dataset Used

This project uses the **Iris dataset** available through Scikit-learn.

The dataset contains three flower classes:

1. Setosa
2. Versicolor
3. Virginica

### Features

The four features are:

* Sepal length
* Sepal width
* Petal length
* Petal width

---

## Libraries Used

The project uses:

```python
NumPy
Matplotlib
Scikit-learn
```

Important Scikit-learn components include:

```python
load_iris()
train_test_split()
StandardScaler()
KNeighborsClassifier()
accuracy_score()
confusion_matrix()
classification_report()
```

---

## Train-Test Split

The dataset is divided into two parts:

```text
80% → Training Data
20% → Testing Data
```

The training data is used to train the model.

The testing data is used to evaluate the model on unseen samples.

---

## Feature Scaling

Feature scaling is important for KNN because KNN uses distance calculations.

The project uses:

```python
StandardScaler()
```

to standardize the features.

Standardization makes the features comparable by transforming them to approximately:

```text
Mean = 0
Standard Deviation = 1
```

Without scaling, features with larger numerical ranges could have a greater effect on the distance calculation.

---

## Euclidean Distance

KNN commonly uses Euclidean distance to calculate how close two points are.

The basic formula is:

```text
d = √((x₁-y₁)² + (x₂-y₂)² + ... + (xₙ-yₙ)²)
```

A smaller distance means that two points are closer together.

---

## Creating the KNN Model

The KNN model is created using:

```python
KNeighborsClassifier(n_neighbors=5)
```

Here:

```text
K = 5
```

means that the model considers the five nearest neighbors.

---

## Model Evaluation

The model is evaluated using several metrics.

### Accuracy

Accuracy measures how many predictions are correct.

```python
accuracy_score(y_test, y_pred)
```

---

### Confusion Matrix

The confusion matrix shows the correct and incorrect classifications for each class.

```python
confusion_matrix(y_test, y_pred)
```

---

### Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

```python
classification_report(y_test, y_pred)
```

---

## Testing Different K Values

The program tests different values of K:

```text
K = 1
K = 3
K = 5
K = 7
K = 9
```

The accuracy of each K value is calculated and displayed.

This demonstrates how changing the K value can affect model performance.

---

## Visualization

The Python program creates a graph showing:

```text
K Value vs Accuracy
```

This helps visualize the relationship between the number of neighbors and the model's test-set accuracy.

---

## New Flower Prediction

The program also predicts the class of a new flower using the following measurements:

```text
Sepal Length = 5.9
Sepal Width  = 3.0
Petal Length = 5.1
Petal Width  = 1.8
```

The trained KNN model predicts which Iris species the new flower most closely resembles.

---

## Key Learning Points

Through this project, I learned:

* What KNN is
* How KNN classification works
* What K represents
* How distance is used in classification
* Euclidean distance
* Train-test splitting
* Feature scaling
* StandardScaler
* KNeighborsClassifier
* Model prediction
* Accuracy
* Confusion matrix
* Classification report
* Comparing different K values
* Predicting new data

---

## AIML Engineer Relevance

KNN is an important foundational machine learning algorithm.

This project builds practical knowledge of:

* Supervised learning
* Classification
* Distance-based algorithms
* Feature scaling
* Model evaluation
* Hyperparameter selection
* Prediction on unseen data

These concepts are useful for developing machine learning solutions in real-world applications.

---

## Project Structure

```text
D12/
│
├── D12-KNN_Classification.py
│
└── D12-KNN_Classification.md
```

---

## How to Run

Install the required libraries:

```bash
pip install numpy matplotlib scikit-learn
```

Run the Python program:

```bash
python D12-KNN_Classification.py
```

---

## Expected Output

The program will display:

* Dataset information
* Training and testing sample counts
* Feature scaling confirmation
* KNN predictions
* Model accuracy
* Confusion matrix
* Classification report
* Accuracy for K = 1, 3, 5, 7, and 9
* Best K value based on test accuracy
* New flower prediction

It will also display a graph of **K Value vs Accuracy**.

---

## Day 12 Completed

**Topic:** K-Nearest Neighbors (KNN) Classification

**Skills:** Python, NumPy, Matplotlib, Scikit-learn, KNN, Feature Scaling, Classification, Model Evaluation

```
```
