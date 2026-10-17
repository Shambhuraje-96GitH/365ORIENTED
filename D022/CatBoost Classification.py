
"""
Day 22: CatBoost Classification
365-Day AIML GitHub Challenge

Objective:
Build and evaluate a CatBoost classification model
using the Titanic dataset available in scikit-learn's
OpenML service.

Note:
The dataset is downloaded from OpenML. An internet
connection is required for the first run.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


def load_dataset():
    """Load the Titanic dataset from OpenML."""
    print("Loading Titanic dataset...")

    data = fetch_openml(
        name="titanic",
        version=1,
        as_frame=True
    )

    df = data.frame.copy()

    # Keep a practical set of commonly used features.
    columns = [
        "pclass", "sex", "age", "sibsp",
        "parch", "fare", "embarked", "survived"
    ]

    df = df[columns].copy()
    df["survived"] = pd.to_numeric(
        df["survived"], errors="coerce"
    )
    df = df.dropna(subset=["survived"])
    df["survived"] = df["survived"].astype(int)

    return df


def prepare_data(df):
    """Separate features and target; split the data."""
    X = df.drop(columns=["survived"])
    y = df["survived"]

    categorical_features = ["sex", "embarked"]
    numerical_features = [
        "pclass", "age", "sibsp", "parch", "fare"
    ]

    # CatBoost supports categorical features natively.
    # Missing categorical values must be converted to strings.
    for column in categorical_features:
        X[column] = X[column].fillna("Unknown").astype(str)

    # Numeric columns are kept numeric; CatBoost can handle
    # missing numeric values.
    for column in numerical_features:
        X[column] = pd.to_numeric(
            X[column], errors="coerce"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return (
        X_train, X_test, y_train, y_test,
        categorical_features
    )


def train_model(X_train, y_train, categorical_features):
    """Train a CatBoost classification model."""
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
        allow_writing_files=False
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features
    )

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    predictions = model.predict(X_test).astype(int).ravel()
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    print("\n========== MODEL EVALUATION ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Not Survived", "Survived"],
            zero_division=0
        )
    )

    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Survived", "Survived"]
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("CatBoost - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # CatBoost provides feature importance directly.
    importance = pd.Series(
        model.feature_importances_,
        index=X_test.columns
    ).sort_values(ascending=True)

    plt.figure(figsize=(9, 5))
    importance.plot(kind="barh")
    plt.title("CatBoost Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    print("\nFeature Importance:")
    print(importance.sort_values(ascending=False))


def main():
    """Run the complete machine learning workflow."""
    df = load_dataset()

    print("\nDataset Preview:")
    print(df.head())

    print("\nDataset Shape:", df.shape)
    print("\nMissing Values:")
    print(df.isnull().sum())

    (
        X_train, X_test, y_train, y_test,
        categorical_features
    ) = prepare_data(df)

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    model = train_model(
        X_train,
        y_train,
        categorical_features
    )

    evaluate_model(model, X_test, y_test)

    # Save the trained model for future inference.
    model.save_model("catboost_titanic_model.cbm")
    print("\nModel saved as catboost_titanic_model.cbm")


if __name__ == "__main__":
    main()
