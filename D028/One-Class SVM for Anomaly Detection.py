"""Day 28 — One-Class SVM for Anomaly Detection

A reproducible, self-contained example of training One-Class SVM on normal
observations only, then evaluating how well it identifies unusual observations.
The example uses synthetic two-feature data; no external dataset is required.

Run:
    python One_Class_SVM_Anomaly_Detection.py

Dependencies:
    numpy, pandas, matplotlib, scikit-learn
"""

from __future__ import annotations

import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM
except ImportError as exc:
    missing_package = getattr(exc, "name", "a required package")
    print(
        f"Missing dependency: {missing_package}. Install dependencies with:\n"
        "python -m pip install numpy pandas matplotlib scikit-learn"
    )
    raise SystemExit(1) from exc


def make_synthetic_data(
    random_state: int = 42,
    n_train_normal: int = 800,
    n_test_normal: int = 250,
    n_test_anomalies: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create normal training data and a labeled test set.

    The model sees only the normal training observations. Labels are created
    only for the test set so that detection performance can be evaluated.
    Label 0 means normal; label 1 means anomaly.
    """
    rng = np.random.default_rng(random_state)

    def sample_normal(count: int) -> np.ndarray:
        return rng.normal(loc=(0.0, 0.0), scale=(1.0, 0.8), size=(count, 2))

    X_train_normal = sample_normal(n_train_normal)
    X_test_normal = sample_normal(n_test_normal)

    # Generate anomalies outside a central radius to make the demonstration
    # meaningfully separable while keeping the points synthetic and reproducible.
    anomaly_chunks: list[np.ndarray] = []
    remaining = n_test_anomalies
    while remaining > 0:
        candidates = rng.uniform(low=-5.0, high=5.0, size=(remaining * 2, 2))
        candidates = candidates[np.linalg.norm(candidates, axis=1) >= 2.5]
        selected = candidates[:remaining]
        if selected.size == 0:
            continue
        anomaly_chunks.append(selected)
        remaining -= len(selected)
    X_test_anomalies = np.vstack(anomaly_chunks)[:n_test_anomalies]

    X_test = np.vstack((X_test_normal, X_test_anomalies))
    y_test = np.concatenate(
        (np.zeros(len(X_test_normal), dtype=int), np.ones(len(X_test_anomalies), dtype=int))
    )
    return X_train_normal, X_test, y_test


def main() -> None:
    print("=" * 68)
    print("DAY 28: ONE-CLASS SVM FOR ANOMALY DETECTION")
    print("=" * 68)

    # 1. Create a reproducible sample dataset.
    X_train_normal, X_test, y_test = make_synthetic_data()
    feature_names = ["Feature_1", "Feature_2"]

    train_df = pd.DataFrame(X_train_normal, columns=feature_names)
    test_df = pd.DataFrame(X_test, columns=feature_names)
    print("\nDataset overview")
    print(f"Normal training observations: {len(train_df)}")
    print(f"Test observations:            {len(test_df)}")
    print(f"  Normal test observations:   {int((y_test == 0).sum())}")
    print(f"  Anomalous test observations:{int((y_test == 1).sum())}")