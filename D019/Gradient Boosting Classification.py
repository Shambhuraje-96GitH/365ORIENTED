
"""
Day 19: Gradient Boosting Classification
365-Day AIML Engineer Challenge

Objective:
Build and evaluate a Gradient Boosting classification model.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def main():
    # 1. Load the dataset
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target

    print("=" * 60)
    print("DAY 19: GRADIENT BOOSTING CLASSIFICATION")
    print("=" * 60)

    print("\nDataset shape:", X.shape)
    print("Number of features:", X.shape[1])
    print("Classes:", dataset.target_names)

    # 2. Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTraining samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    # 3. Build the model pipeline
    # Scaling is included for a consistent ML workflow.
    # Gradient Boosting trees do not require feature scaling.
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
            ),
        ),
    ])

    # 4. Train the model
    model.fit(X_train, y_train)
    print("\nModel training completed!")

    # 5. Make predictions
    y_pred = model.predict(X_test)

    # 6. Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=dataset.target_names,
        )
    )

    # 7. Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=dataset.target_names,
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Gradient Boosting - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("gradient_boosting_confusion_matrix.png", dpi=300)
    plt.show()

    # 8. Plot feature importance
    classifier = model.named_steps["classifier"]
    importances = classifier.feature_importances_

    top_indices = np.argsort(importances)[-10:]
    top_importances = importances[top_indices]
    top_features = dataset.feature_names[top_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(top_features, top_importances)
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Top 10 Important Features")
    plt.tight_layout()
    plt.savefig("gradient_boosting_feature_importance.png", dpi=300)
    plt.show()

    # 9. Predict a sample
    sample = X_test[0].reshape(1, -1)
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    print("\nSample Prediction:")
    print("Predicted class:", dataset.target_names[prediction])
    print("Class probabilities:")

    for class_name, probability in zip(
        dataset.target_names, probabilities
    ):
        print(f"  {class_name}: {probability:.4f}")

    print("\nDay 19 completed successfully!")


if __name__ == "__main__":
    main()
