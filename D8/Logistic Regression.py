# Day 8: Logistic Regression
# Goal: Predict whether a student passes or fails

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -----------------------------------
# 1. Create dataset
# -----------------------------------

data = {
    "StudyHours": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9],
    "Attendance": [50, 55, 60, 62, 65, 68, 70, 75, 78, 80, 82, 85, 90, 95],
    "Passed": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# -----------------------------------
# 2. Separate features and target
# -----------------------------------

X = df[["StudyHours", "Attendance"]]
y = df["Passed"]

# -----------------------------------
# 3. Split data
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# -----------------------------------
# 4. Create Logistic Regression model
# -----------------------------------

model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# -----------------------------------
# 5. Make predictions
# -----------------------------------

y_pred = model.predict(X_test)

print("\nActual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_pred)

# -----------------------------------
# 6. Accuracy
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# -----------------------------------
# 7. Confusion Matrix
# -----------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# -----------------------------------
# 8. Classification Report
# -----------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------------
# 9. Predict a new student
# -----------------------------------

new_student = [[6, 85]]

prediction = model.predict(new_student)

print("\nNew Student Prediction:")

if prediction[0] == 1:
    print("Student is predicted to PASS")
else:
    print("Student is predicted to FAIL")

# Probability
probability = model.predict_proba(new_student)

print("\nPrediction Probability:")
print(probability)