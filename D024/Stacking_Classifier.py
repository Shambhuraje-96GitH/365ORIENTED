"""
Day 24 — Stacking Classifier (Advanced Ensemble Learning)
365-Day AIML Engineer GitHub Challenge

Overview
--------
Stacking combines predictions from multiple base models and trains a final
model (the meta-model) to learn how best to combine them.

This educational example uses scikit-learn's built-in Breast Cancer Wisconsin
Dataset. It compares Logistic Regression, Random Forest, SVC, and a
Stacking Classifier using common classification metrics.

Note: This is a learning exercise, not a medical diagnostic tool.

Install dependencies:
    python -m pip install scikit-learn pandas matplotlib

Run:
    python Stacking_Classifier.py
"""

import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42


def load_data():
    """Load the built-in dataset and return features, target, and class names."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    return X, y, dataset.target_names


def build_models():
    """Create individual estimators and a stacking ensemble."""
    logistic_regression = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    support_vector_classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # The final estimator learns from out-of-fold predictions of the base models.
    stacking_classifier = StackingClassifier(
        estimators=[
            ("logistic_regression", logistic_regression),
            ("random_forest", random_forest),
            ("svc", support_vector_classifier),
        ],
        final_estimator=LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )

    return {
        "Logistic Regression": logistic_regression,
        "Random Forest": random_forest,
        "Support Vector Classifier": support_vector_classifier,
        "Stacking Classifier": stacking_classifier,
    }


def evaluate_model(name, model, X_test, y_test):
    """Calculate standard binary-classification evaluation metrics."""
    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        positive_probabilities = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, positive_probabilities)
    else:
        roc_auc = float("nan")

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc,
    }
    return metrics, predictions


def main():
    """Run the complete stacking-classification workflow."""
    warnings.filterwarnings("ignore", category=UserWarning)

    print("=" * 72)
    print("DAY 24: STACKING CLASSIFIER — ADVANCED ENSEMBLE LEARNING")
    print("=" * 72)

    # 1. Load data
    X, y, target_names = load_data()
    print(f"\nDataset shape: {X.shape[0]} rows × {X.shape[1]} features")
    print("Target classes:", ", ".join(target_names))
    print("Class distribution:")
    print(y.map({0: target_names[0], 1: target_names[1]}).value_counts())

    # 2. Create a stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # 3. Build and train models
    models = build_models()
    results = []
    stacking_predictions = None
    stacking_model = None

    print("\nTraining models...")
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        metrics, predictions = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)

        if name == "Stacking Classifier":
            stacking_predictions = predictions
            stacking_model = model

    # 4. Compare model metrics
    results_df = pd.DataFrame(results).set_index("Model")
    print("\nMODEL PERFORMANCE COMPARISON")
    print(results_df.round(4).to_string())

    # 5. Detailed evaluation of the stacking model
    print("\nSTACKING CLASSIFIER — CLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            stacking_predictions,
            target_names=target_names,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_test, stacking_predictions)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(matrix)

    best_model_name = results_df["F1 Score"].idxmax()
    print(f"\nHighest F1 score in this run: {best_model_name}")
    print(
        "Remember: results depend on the dataset and split. Stacking is not "
        "guaranteed to outperform every base model."
    )

    # 6. Plot the metric comparison
    ax = results_df[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]].plot(
        kind="bar",
        figsize=(12, 6),
        ylim=(0.0, 1.05),
        rot=15,
        title="Day 24 — Classification Model Comparison",
    )
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.show()

    # 7. Plot the stacking model confusion matrix
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=target_names,
    ).plot(values_format="d")
    plt.title("Stacking Classifier — Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Keep a reference for clarity and future extensions.
    assert stacking_model is not None
    print("\nDay 24 completed successfully.")
    print("Next step: document the workflow and findings in your README.md.")


if __name__ == "__main__":
    main()
