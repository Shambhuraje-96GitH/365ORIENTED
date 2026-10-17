import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_dataset(random_state: int = 42):
    """Create a two-feature dataset with normal points and injected anomalies."""
    rng = np.random.default_rng(random_state)

    normal_count = 1000
    anomaly_count = 60

    normal_data = rng.normal(loc=0.0, scale=1.0, size=(normal_count, 2))
    anomaly_data = rng.uniform(low=-6.0, high=6.0, size=(anomaly_count, 2))

    X = np.vstack((normal_data, anomaly_data))
    # 0 = normal, 1 = anomaly
    y = np.concatenate(
        (np.zeros(normal_count, dtype=int), np.ones(anomaly_count, dtype=int))
    )

    # Shuffle observations while keeping labels aligned.
    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]

    features = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])
    return features, y


def main():
    print("=" * 65)
    print("DAY 26: ISOLATION FOREST FOR ANOMALY DETECTION")
    print("=" * 65)

    X, y = create_dataset()
    print(f"\nDataset shape: {X.shape}")
    print(f"Normal observations: {(y == 0).sum()}")
    print(f"Anomalies injected:  {(y == 1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Fit scaling only on the training data to avoid data leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Isolation Forest returns -1 for anomalies and 1 for normal observations.
    model = IsolationForest(
        n_estimators=200,
        contamination=0.06,
        random_state=42,
    )
    model.fit(X_train_scaled)

    raw_predictions = model.predict(X_test_scaled)
    # Convert predictions to conventional labels: 0 = normal, 1 = anomaly.
    y_pred = np.where(raw_predictions == -1, 1, 0)

    print("\n--- Evaluation on the held-out test set ---")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")