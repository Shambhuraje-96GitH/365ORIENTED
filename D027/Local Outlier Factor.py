
"""
Day 27: Local Outlier Factor (LOF) for Anomaly Detection
365-Day AIML Engineer GitHub Challenge

Description:
Detect anomalous data points using the Local Outlier Factor (LOF)
algorithm from scikit-learn.

Technologies:
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
# 1. Generate a Sample Dataset
# ============================================================

def generate_dataset(random_state=42):
    """
    Generate normal observations and artificial anomalies.

    Returns:
        X: Feature matrix
        y_true: Ground-truth labels
               1 = Normal, -1 = Anomaly
    """

    # Generate normal observations in two-dimensional space
    X_normal, _ = make_blobs(
        n_samples=500,
        centers=1,
        n_features=2,
        cluster_std=1.0,
        center_box=(-2.0, 2.0),
        random_state=random_state
    )

    # Generate anomalies far from the normal observations
    rng = np.random.default_rng(random_state)

    X_anomalies = rng.uniform(
        low=-8,
        high=8,
        size=(30, 2)
    )

    # Combine normal observations and anomalies
    X = np.vstack([X_normal, X_anomalies])

    # Ground-truth labels: 1 = normal, -1 = anomaly
    y_true = np.concatenate([
        np.ones(len(X_normal), dtype=int),
        -np.ones(len(X_anomalies), dtype=int)
    ])

    return X, y_true


# ============================================================
# 2. Preprocess the Dataset
# ============================================================

def preprocess_data(X):
    """
    Standardize the input features.

    LOF relies on distances between observations, so feature
    scaling is important when features have different scales.
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


# ============================================================
# 3. Train the LOF Anomaly Detection Model
# ============================================================

def detect_anomalies(X_scaled, n_neighbors=20, contamination=0.05):
    """
    Fit the Local Outlier Factor model.

    Parameters:
        X_scaled: Standardized feature matrix
        n_neighbors: Number of neighbors used for local density
        contamination: Expected proportion of anomalies

    Returns:
        model: Trained LOF model
        predictions: 1 for normal, -1 for anomaly
    """

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )

    predictions = model.fit_predict(X_scaled)

    return model, predictions


# ============================================================
# 4. Evaluate the Model
# ============================================================

def evaluate_model(y_true, predictions):
    """
    Evaluate predictions against known synthetic labels.
    """

    # Convert labels into binary format for evaluation:
    # 1 = anomaly, 0 = normal
    y_true_binary = (y_true == -1).astype(int)
    predictions_binary = (predictions == -1).astype(int)

    precision = precision_score(
        y_true_binary,
        predictions_binary,
        zero_division=0
    )

    recall = recall_score(
        y_true_binary,
        predictions_binary,
        zero_division=0
    )

    f1 = f1_score(
        y_true_binary,
        predictions_binary,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("LOCAL OUTLIER FACTOR — EVALUATION RESULTS")
    print("=" * 60)

    print(f"Total observations : {len(y_true)}")
    print(f"Actual anomalies   : {np.sum(y_true == -1)}")
    print(f"Detected anomalies : {np.sum(predictions == -1)}")
    print(f"Detected normal    : {np.sum(predictions == 1)}")

    print("\nConfusion Matrix")
    print("Rows = Actual [Normal, Anomaly]")
    print("Cols = Predicted [Normal, Anomaly]")
    print(confusion_matrix(
        y_true_binary,
        predictions_binary,
        labels=[0, 1]
    ))

    print("\nClassification Report")
    print(classification_report(
        y_true_binary,
        predictions_binary,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0
    ))

    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")


# ============================================================
# 5. Visualize Detected Anomalies
# ============================================================

def visualize_results(X, predictions, y_true):
    """
    Plot the detected anomalies and compare them with
    the known synthetic ground truth.
    """

    detected_normal = predictions == 1
    detected_anomalies = predictions == -1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot LOF predictions
    axes[0].scatter(
        X[detected_normal, 0],
        X[detected_normal, 1],
        c="steelblue",
        label="Predicted Normal",
        alpha=0.7,
        s=35
    )

    axes[0].scatter(
        X[detected_anomalies, 0],
        X[detected_anomalies, 1],
        c="red",
        label="Predicted Anomaly",
        marker="x",
        s=70
    )

    axes[0].set_title("LOF Anomaly Detection")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # Plot actual synthetic labels
    actual_normal = y_true == 1
    actual_anomalies = y_true == -1

    axes[1].scatter(
        X[actual_normal, 0],
        X[actual_normal, 1],
        c="steelblue",
        label="Actual Normal",
        alpha=0.7,
        s=35
    )

    axes[1].scatter(
        X[actual_anomalies, 0],
        X[actual_anomalies, 1],
        c="orange",
        label="Actual Anomaly",
        marker="x",
        s=70
    )

    axes[1].set_title("Ground-Truth Labels")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


# ============================================================
# 6. Display the Most Anomalous Observations
# ============================================================

def display_anomalies(X, predictions, model, count=10):
    """
    Display detected anomalies with their LOF scores.

    More negative LOF scores generally indicate observations
    that are more anomalous relative to their neighbors.
    """

    anomaly_indices = np.where(predictions == -1)[0]

    if len(anomaly_indices) == 0:
        print("\nNo anomalies detected.")
        return

    # negative_outlier_factor_ contains scores for training data
    scores = model.negative_outlier_factor_[anomaly_indices]

    anomaly_table = pd.DataFrame({
        "Index": anomaly_indices,
        "Feature_1": X[anomaly_indices, 0],
        "Feature_2": X[anomaly_indices, 1],
        "LOF_Score": scores
    })

    # The most negative scores appear first
    anomaly_table = anomaly_table.sort_values("LOF_Score")

    print(f"\nTop {min(count, len(anomaly_table))} Most Anomalous Observations")
    print("-" * 60)
    print(anomaly_table.head(count).to_string(index=False))


# ============================================================
# 7. Main Execution
# ============================================================

def main():
    print("=" * 60)
    print("DAY 27: LOCAL OUTLIER FACTOR (LOF)")
    print("Unsupervised Anomaly Detection")
    print("=" * 60)

    # Step 1: Generate data
    X, y_true = generate_dataset()

    print(f"\nDataset shape: {X.shape}")
    print(f"Normal observations: {np.sum(y_true == 1)}")
    print(f"Actual anomalies: {np.sum(y_true == -1)}")

    # Step 2: Standardize features
    X_scaled, _ = preprocess_data(X)

    # Step 3: Detect anomalies
    model, predictions = detect_anomalies(
        X_scaled,
        n_neighbors=20,
        contamination=0.05
    )

    # Step 4: Evaluate detection performance
    evaluate_model(y_true, predictions)

    # Step 5: Show the most anomalous observations
    display_anomalies(
        X,
        predictions,
        model,
        count=10
    )

    # Step 6: Visualize results
    visualize_results(X, predictions, y_true)

    print("\nDay 27 execution completed successfully!")


if __name__ == "__main__":
    main()
