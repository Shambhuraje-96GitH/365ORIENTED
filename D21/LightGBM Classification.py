"""
Day 21 - LightGBM Classification
365-Day AIML Engineer GitHub Challenge

Description:
    Train and evaluate a LightGBM classifier on scikit-learn's Breast Cancer
    Wisconsin dataset. The script reports common classification metrics,
    visualizes model performance and feature importance, and saves the trained
    model for later inference.

Install dependencies:
    python -m pip install lightgbm scikit-learn pandas numpy matplotlib seaborn joblib

Run:
    python LightGBM_Classification.py

Generated files (created in the current working directory):
    lightgbm_breast_cancer_model.joblib
    lightgbm_confusion_matrix.png
    lightgbm_roc_curve.png
    lightgbm_feature_importance.png
"""

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
OUTPUT_DIR = Path(".")
MODEL_PATH = OUTPUT_DIR / "lightgbm_breast_cancer_model.joblib"


def load_dataset():
    """Load the example dataset and return features, target, and class names."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    target_names = list(dataset.target_names)

    print("=" * 72)
    print("DAY 21: LIGHTGBM CLASSIFICATION")
    print("=" * 72)
    print(f"Dataset shape: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"Class labels: {dict(enumerate(target_names))}")
    print("\nClass distribution:")
    print(y.value_counts().sort_index().rename(index=dict(enumerate(target_names))))

    return X, y, target_names


def train_model(X_train, y_train):
    """Create and train a LightGBM binary classifier."""
    model = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, target_names):
    """Print evaluation metrics and return predictions and probabilities."""
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probability)

    print("\n" + "=" * 72)
    print("MODEL EVALUATION ON THE HELD-OUT TEST SET")
    print("=" * 72)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    return y_pred, y_probability


def save_confusion_matrix(y_test, y_pred, target_names):
    """Create and save a confusion-matrix heatmap."""
    matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("LightGBM - Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    path = OUTPUT_DIR / "lightgbm_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix: {path.resolve()}")


def save_roc_curve(y_test, y_probability):
    """Create and save an ROC curve for the positive class."""
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, y_probability, ax=ax)
    ax.set_title("LightGBM - ROC Curve")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUTPUT_DIR / "lightgbm_roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curve: {path.resolve()}")


def save_feature_importance(model, feature_names):
    """Plot and save the 15 most important features."""
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False).head(15).sort_values()

    plt.figure(figsize=(9, 6))
    importance.plot(kind="barh")
    plt.title("LightGBM - Top 15 Feature Importances")
    plt.xlabel("Importance (number of splits)")
    plt.ylabel("Feature")
    plt.tight_layout()
    path = OUTPUT_DIR / "lightgbm_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved feature-importance chart: {path.resolve()}")

    print("\nTop 10 features by importance:")
    print(importance.sort_values(ascending=False).head(10).to_string())


def demonstrate_single_prediction(model, X_test, target_names):
    """Show a sample prediction and class probabilities."""
    sample = X_test.iloc[[0]]
    predicted_class = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]

    print("\n" + "=" * 72)
    print("SAMPLE PREDICTION")
    print("=" * 72)
    print(f"Predicted class: {predicted_class} ({target_names[predicted_class]})")
    print("Predicted probabilities:")
    for class_id, class_name in enumerate(target_names):
        print(f"  {class_name}: {probabilities[class_id]:.4f}")


def main():
    """Run the full Day 21 workflow."""
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

    try:
        X, y, target_names = load_dataset()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        print(f"\nTraining samples: {len(X_train)}")
        print(f"Testing samples : {len(X_test)}")
        print("\nTraining LightGBM classifier...")
        model = train_model(X_train, y_train)
        print("Training completed.")

        y_pred, y_probability = evaluate_model(model, X_test, y_test, target_names)

        save_confusion_matrix(y_test, y_pred, target_names)
        save_roc_curve(y_test, y_probability)
        save_feature_importance(model, X.columns)
        demonstrate_single_prediction(model, X_test, target_names)

        joblib.dump(model, MODEL_PATH)
        print(f"\nSaved trained model: {MODEL_PATH.resolve()}")
        print("\nDay 21 complete! Review the metrics and generated charts before committing.")

    except ImportError as error:
        print("A required package is missing.")
        print("Install dependencies with:")
        print("python -m pip install lightgbm scikit-learn pandas numpy matplotlib seaborn joblib")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
