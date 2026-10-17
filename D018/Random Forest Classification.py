
"""
Day 18: Random Forest Classification
365-Day AIML Engineer GitHub Challenge

Description:
    Build a Random Forest Classification model using the
    Breast Cancer Wisconsin dataset.

Topics covered:
    1. Dataset loading and exploration
    2. Train-test splitting
    3. Random Forest model training
    4. Classification predictions
    5. Model evaluation
    6. Confusion matrix visualization
    7. Feature importance visualization
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 100
TOP_FEATURES = 10


# --------------------------------------------------
# 2. Load Dataset
# --------------------------------------------------

def load_dataset():
    """Load the Breast Cancer Wisconsin dataset."""

    dataset = load_breast_cancer()

    X = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names
    )

    y = pd.Series(
        dataset.target,
        name="target"
    )

    target_names = dataset.target_names

    print("=" * 65)
    print("DAY 18: RANDOM FOREST CLASSIFICATION")
    print("=" * 65)

    print("\nDataset loaded successfully!")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Target classes: {list(target_names)}")

    return X, y, target_names


# --------------------------------------------------
# 3. Explore Dataset
# --------------------------------------------------

def explore_dataset(X, y, target_names):
    """Display basic dataset information."""

    print("\n" + "=" * 65)
    print("DATASET EXPLORATION")
    print("=" * 65)

    print("\nFirst five rows:")
    print(X.head())

    print("\nDataset information:")
    print(f"Dataset shape: {X.shape}")

    print("\nMissing values:")
    print(f"Total missing values: {X.isnull().sum().sum()}")

    print("\nClass distribution:")
    for class_id, class_name in enumerate(target_names):
        count = int((y == class_id).sum())
        print(f"{class_name}: {count} samples")

    print("\nStatistical summary:")
    print(X.describe().round(2))


# --------------------------------------------------
# 4. Split Dataset
# --------------------------------------------------

def split_dataset(X, y):
    """Split the data into training and testing sets."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n" + "=" * 65)
    print("TRAIN-TEST SPLIT")
    print("=" * 65)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# 5. Train Random Forest Model
# --------------------------------------------------

def train_model(X_train, y_train):
    """Train the Random Forest classifier."""

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("\n" + "=" * 65)
    print("MODEL TRAINING")
    print("=" * 65)

    model.fit(X_train, y_train)

    print("Random Forest model trained successfully!")
    print(f"Number of decision trees: {N_ESTIMATORS}")

    return model


# --------------------------------------------------
# 6. Make Predictions
# --------------------------------------------------

def make_predictions(model, X_test):
    """Generate predictions and class probabilities."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print("\n" + "=" * 65)
    print("SAMPLE PREDICTIONS")
    print("=" * 65)

    print("\nFirst 10 predictions:")
    print(predictions[:10])

    print("\nClass probabilities for the first 5 samples:")
    print(np.round(probabilities[:5], 3))

    return predictions


# --------------------------------------------------
# 7. Evaluate Model
# --------------------------------------------------

def evaluate_model(model, X_test, y_test, predictions, target_names):
    """Calculate and display classification metrics."""

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\n" + "=" * 65)
    print("MODEL EVALUATION")
    print("=" * 65)

    print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\nDetailed Classification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=target_names,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(target_names))
    )

    print("Confusion Matrix:")
    print(cm)

    return cm


# --------------------------------------------------
# 8. Visualize Confusion Matrix
# --------------------------------------------------

def plot_confusion_matrix(cm, target_names):
    """Plot the confusion matrix using Seaborn."""

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )

    plt.title("Random Forest - Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 9. Visualize Feature Importance
# --------------------------------------------------

def plot_feature_importance(model, feature_names):
    """Display the top features used by the model."""

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    top_features = importance_df.head(TOP_FEATURES)

    print("\n" + "=" * 65)
    print(f"TOP {TOP_FEATURES} IMPORTANT FEATURES")
    print("=" * 65)

    print(top_features.to_string(index=False))

    plt.figure(figsize=(10, 7))

    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="viridis",
        legend=False
    )

    plt.title(f"Top {TOP_FEATURES} Random Forest Features")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    return importance_df


# --------------------------------------------------
# 10. Predict a Single Sample
# --------------------------------------------------

def predict_single_sample(model, X_test, target_names):
    """Predict the class of one held-out test sample."""

    sample = X_test.iloc[[0]]

    predicted_class = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    print("\n" + "=" * 65)
    print("SINGLE SAMPLE PREDICTION")
    print("=" * 65)

    print(f"Predicted class: {target_names[predicted_class]}")

    print("\nClass probabilities:")
    for class_id, class_name in enumerate(target_names):
        print(
            f"{class_name}: "
            f"{probabilities[class_id] * 100:.2f}%"
        )

    print("\nNote: This is a demonstration prediction,")
    print("not a substitute for professional medical diagnosis.")


# --------------------------------------------------
# 11. Main Function
# --------------------------------------------------

def main():
    """Run the complete Random Forest classification workflow."""

    # Load data
    X, y, target_names = load_dataset()

    # Explore data
    explore_dataset(X, y, target_names)

    # Split data
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Train model
    model = train_model(X_train, y_train)

    # Make predictions
    predictions = make_predictions(model, X_test)

    # Evaluate model
    cm = evaluate_model(
        model,
        X_test,
        y_test,
        predictions,
        target_names
    )

    # Plot confusion matrix
    plot_confusion_matrix(cm, target_names)

    # Analyze feature importance
    plot_feature_importance(model, X.columns)

    # Predict a single sample
    predict_single_sample(model, X_test, target_names)

    print("\n" + "=" * 65)
    print("DAY 18 COMPLETED SUCCESSFULLY!")
    print("=" * 65)


# --------------------------------------------------
# 12. Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    main()