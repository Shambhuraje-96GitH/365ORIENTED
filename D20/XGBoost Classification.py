
"""
Day 20: XGBoost Classification
365-Day AIML GitHub Challenge

Project:
Binary classification using the Breast Cancer dataset.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)


def main():
    # 1. Load the dataset
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target

    print("=" * 60)
    print("DAY 20: XGBOOST CLASSIFICATION")
    print("=" * 60)

    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Target classes: {dataset.target_names}")

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # 3. Create the XGBoost model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # 4. Train the model
    model.fit(X_train, y_train)
    print("\nModel training completed successfully!")

    # 5. Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 6. Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\nMODEL EVALUATION")
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")

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

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=dataset.target_names,
    ).plot(cmap="Blues", values_format="d")

    plt.title("XGBoost - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # 8. Plot feature importance
    importance = model.feature_importances_
    top_indices = importance.argsort()[-10:]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=importance[top_indices],
        y=[dataset.feature_names[i] for i in top_indices],
    )

    plt.title("Top 10 Important Features - XGBoost")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    print("\nTop 10 Important Features:")
    for index in importance.argsort()[-10:][::-1]:
        print(
            f"{dataset.feature_names[index]}: "
            f"{importance[index]:.4f}"
        )


if __name__ == "__main__":
    main()
