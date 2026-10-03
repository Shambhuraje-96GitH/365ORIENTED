# Day 13 — Support Vector Machine (SVM) Classification

## 🎯 Objective

The objective of Day 13 is to understand and implement the **Support Vector Machine (SVM)** classification algorithm using Python and Scikit-learn.

The **Iris dataset** is used to train and evaluate the SVM classification model.

---

## 🧠 What is SVM?

Support Vector Machine (SVM) is a supervised machine learning algorithm that can be used for classification and regression.

For classification, SVM attempts to find a decision boundary, called a **hyperplane**, that separates different classes of data.

The data points closest to the decision boundary are called **support vectors**.

---

## ⚙️ How SVM Works

The basic idea of SVM is:

1. Load the training data.
2. Identify different classes.
3. Find a suitable decision boundary.
4. Maximize the margin between classes.
5. Use support vectors to define the boundary.
6. Use the trained boundary to classify new data.

A larger margin generally provides a more robust separation between classes.

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

The dataset contains:

* **150 samples**
* **4 numerical features**
* **3 target classes**

---

## 🔀 Train-Test Split

The dataset is divided into training and testing data.

The project uses:

```text
80% → Training data
20% → Testing data
```

The training data is used to train the SVM model, while the testing data is used to evaluate its predictions.

---

## 📏 Feature Scaling

Feature scaling is important for SVM because the algorithm uses distances and margins when determining the decision boundary.

`StandardScaler` is used to standardize the features.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then applied to the testing data.

---

## 🤖 Creating the SVM Model

The initial SVM model uses the **RBF kernel**.

```python
svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)
```

The model is then trained:

```python
svm_model.fit(X_train_scaled, y_train)
```

---

## 🔮 Making Predictions

After training, the model predicts the classes of the test data:

```python
y_pred = svm_model.predict(X_test_scaled)
```

---

## 📈 Model Evaluation

### Accuracy

Accuracy measures the percentage of correctly classified samples.

```python
accuracy_score(y_test, y_pred)
```

---

### Confusion Matrix

A confusion matrix shows the number of correct and incorrect predictions for each class.

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
classification_report(
    y_test,
    y_pred,
    target_names=target_names
)
```

---

## 🔬 SVM Kernels

The project compares three commonly used SVM kernels:

```text
Linear
RBF
Polynomial
```

### 1. Linear Kernel

The linear kernel is useful when the classes can be separated using a linear decision boundary.

```python
SVC(kernel="linear")
```

### 2. RBF Kernel

The RBF (Radial Basis Function) kernel can model nonlinear relationships between data points.

```python
SVC(kernel="rbf")
```

### 3. Polynomial Kernel

The polynomial kernel creates a decision boundary based on polynomial relationships.

```python
SVC(kernel="poly")
```

---

## 📊 Kernel Comparison

The program compares the accuracy of each kernel.

Example output format:

```text
Kernel = linear --> Accuracy = ...
Kernel = rbf    --> Accuracy = ...
Kernel = poly   --> Accuracy = ...
```

The exact accuracy values depend on the dataset split and model configuration.

---

## 🧪 Parameters Used

The main RBF SVM model uses:

```text
kernel = rbf
C = 1.0
gamma = scale
```

### C Parameter

`C` controls the trade-off between having a wider margin and correctly classifying training examples.

### Gamma Parameter

For the RBF kernel, `gamma` controls how strongly individual training examples influence the decision boundary.

---

## 📉 Visualization

The Python program creates a bar chart comparing:

```text
SVM Kernel → Accuracy
```

This provides a simple visual comparison of the tested kernels.

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
├── Day 13 — SVM Classification.md
└── SVM Classification.py
```

---

## 💡 Key Learnings

From Day 13, I learned:

* What Support Vector Machines are
* What a hyperplane is
* What support vectors are
* How SVM classification works
* Why margins are important
* Why feature scaling is useful for SVM
* How to train an SVM classifier
* How to make predictions
* How to calculate accuracy
* How to use a confusion matrix
* How to generate a classification report
* What SVM kernels are
* How Linear, RBF, and Polynomial kernels differ
* How to compare different SVM kernels

---

## 🚀 Day 13 Challenge

Experiment with different values of `C` and observe how the model performance changes.

Try:

```text
C = 0.1
C = 1
C = 10
C = 100
```

Also experiment with the RBF `gamma` parameter.

The goal is to understand how SVM hyperparameters influence the decision boundary and model performance.

---

## 📌 Conclusion

SVM is an important supervised machine learning algorithm for classification.

This project followed the machine learning workflow:

```text
Load Dataset
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
SVM Model
      ↓
Prediction
      ↓
Evaluation
      ↓
Kernel Comparison
```

Day 13 provided practical experience with SVM classification and introduced important concepts such as **hyperplanes, margins, support vectors, kernels, and model evaluation**.
