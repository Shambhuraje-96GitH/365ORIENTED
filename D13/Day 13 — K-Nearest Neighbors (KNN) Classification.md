# Day 13 — K-Nearest Neighbors (KNN) Classification

## 🎯 Objective

The objective of Day 13 is to understand and implement the **K-Nearest Neighbors (KNN)** classification algorithm using Python and Scikit-learn.

The Iris dataset is used to train and evaluate the classification model.

---

## 🧠 What is KNN?

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression problems.

In classification, KNN predicts the class of a new data point by looking at the classes of its nearest neighboring data points.

The value of **K** represents the number of neighbors considered when making a prediction.

For example, if:

```text
K = 3
```

the algorithm looks at the three closest data points and uses their classes to determine the prediction.

---

## ⚙️ How KNN Works

The basic steps of KNN are:

1. Choose the value of K.
2. Calculate the distance between the new data point and existing training points.
3. Find the K nearest data points.
4. Check the classes of those neighbors.
5. Assign the most common class to the new data point.

---

## 📊 Dataset Used

The **Iris dataset** is used for this project.

It contains measurements of iris flowers.

### Features

* Sepal length
* Sepal width
* Petal length
* Petal width

### Target Classes

* Setosa
* Versicolor
* Virginica

The dataset contains **150 samples** and **4 numerical features**.

---

## 🔀 Train-Test Split

The dataset is divided into:

* 80% training data
* 20% testing data

The training data is used to train the KNN model, while the testing data is used to evaluate its performance.

---

## 📏 Feature Scaling

Feature scaling is important for KNN because KNN uses distance calculations.

If one feature has much larger values than another feature, it can have a greater influence on the distance calculation.

`StandardScaler` is therefore used to standardize the features.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then applied to the testing data.

---

## 🤖 Creating the KNN Model

The initial model uses:

```python
K = 5
```

The model is created using:

```python
knn = KNeighborsClassifier(n_neighbors=5)
```

Then the model is trained:

```python
knn.fit(X_train_scaled, y_train)
```

---

## 🔮 Making Predictions

After training, predictions are made using:

```python
y_pred = knn.predict(X_test_scaled)
```

---

## 📈 Model Evaluation

### Accuracy

Accuracy measures the proportion of predictions that are correct.

```python
accuracy_score(y_test, y_pred)
```

---

### Confusion Matrix

A confusion matrix shows how many samples were correctly and incorrectly classified for each class.

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

It can be generated using:

```python
classification_report(y_test, y_pred)
```

---

## 🔬 Experimenting with Different K Values

Different K values are tested:

```text
K = 1
K = 3
K = 5
K = 7
K = 9
```

The program compares the accuracy obtained for each value.

Example output:

```text
K = 1 --> Accuracy = ...
K = 3 --> Accuracy = ...
K = 5 --> Accuracy = ...
K = 7 --> Accuracy = ...
K = 9 --> Accuracy = ...
```

The exact values may vary depending on the train-test split and dataset configuration.

---

## ⚖️ Effect of K

The value of K can affect the behavior of a KNN model.

### Small K

A small K makes the model more sensitive to nearby individual data points.

It can make the model more sensitive to noise.

### Large K

A larger K considers more neighboring points.

This can make predictions smoother, but if K becomes too large, local patterns can be lost.

Therefore, selecting an appropriate K value is an important part of using KNN.

---

## 📊 Visualization

The Python program generates a graph showing:

```text
K Value vs Accuracy
```

This helps visualize how model accuracy changes when different K values are used.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## 📂 Project Structure

```text
D13/
│
├── Day13_KNN_Classification.py
└── Day13_KNN_Classification.md
```

---

## 💡 Key Learnings

From Day 13, I learned:

* What KNN classification is
* How KNN uses neighboring data points
* The meaning of K
* Why distance matters in KNN
* Why feature scaling is important
* How to train a KNN classifier
* How to make predictions
* How to calculate accuracy
* How to use a confusion matrix
* How to generate a classification report
* How different K values affect model performance

---

## 🚀 Day 13 Challenge

As an additional practice task, experiment with different K values and observe how the accuracy changes.

Try:

```text
K = 1
K = 3
K = 5
K = 7
K = 9
K = 11
K = 15
```

Record your observations and understand why changing K can change the model's predictions.

---

## 📌 Conclusion

K-Nearest Neighbors is a simple but important machine learning classification algorithm.

This exercise provided practical experience with:

**Data → Train/Test Split → Feature Scaling → KNN Model → Prediction → Evaluation → K Experimentation**

This forms another important step toward building practical machine learning skills.
