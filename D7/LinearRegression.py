# ============================================================
# Day 7 — Introduction to Machine Learning
# Topic: Linear Regression using Scikit-Learn
# ============================================================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------
# Task 1: Create Dataset
# ------------------------------------------------------------

# Feature: Study hours
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)

# Target: Exam scores
y = np.array([35, 40, 45, 50, 55, 60, 65, 72, 78, 85])


print("Study Hours:")
print(X)

print("\nExam Scores:")
print(y)


# ------------------------------------------------------------
# Task 2: Split Data into Training and Testing Sets
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# Task 3: Create Linear Regression Model
# ------------------------------------------------------------

model = LinearRegression()


# ------------------------------------------------------------
# Task 4: Train the Model
# ------------------------------------------------------------

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# ------------------------------------------------------------
# Task 5: Make Predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test)

print("\nActual Values:")
print(y_test)

print("\nPredicted Values:")
print(y_pred)


# ------------------------------------------------------------
# Task 6: Evaluate the Model
# ------------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------------------")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# ------------------------------------------------------------
# Task 7: Inspect Model Parameters
# ------------------------------------------------------------

print("\nModel Parameters")
print("----------------------------")
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)


# ------------------------------------------------------------
# Task 8: Make a New Prediction
# ------------------------------------------------------------

hours = np.array([[7.5]])

prediction = model.predict(hours)

print("\nPrediction")
print("----------------------------")
print("Study Hours:", hours[0][0])
print("Predicted Score:", prediction[0])


# ------------------------------------------------------------
# Task 9: Extra Challenge
# ------------------------------------------------------------

# Try changing the study hours below
new_hours = np.array([[12]])

new_prediction = model.predict(new_hours)

print("\nExtra Prediction")
print("----------------------------")
print("Study Hours:", new_hours[0][0])
print("Predicted Score:", new_prediction[0])


# ============================================================
# End of Day 7
# ============================================================