
"""
Day 30: HDBSCAN Clustering
365-Day AIML Engineer Challenge

Algorithm:
Hierarchical Density-Based Spatial Clustering of
Applications with Noise (HDBSCAN)
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score


def create_dataset():
    """Generate clusters with different densities and random noise."""

    X1, _ = make_blobs(
        n_samples=300,
        centers=[(-5, -2)],
        cluster_std=0.45,
        random_state=42,
    )

    X2, _ = make_blobs(
        n_samples=400,
        centers=[(0, 0)],
        cluster_std=1.0,
        random_state=42,
    )

    X3, _ = make_blobs(
        n_samples=250,
        centers=[(5, 3)],
        cluster_std=0.65,
        random_state=42,
    )

    rng = np.random.default_rng(42)
    noise = rng.uniform(-9, 9, size=(80, 2))

    X = np.vstack([X1, X2, X3, noise])

    return X


def train_hdbscan(X):
    """Scale features and train the HDBSCAN model."""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = HDBSCAN(
        min_cluster_size=25,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    labels = model.fit_predict(X_scaled)

    return model, scaler, X_scaled, labels


def evaluate_clustering(X_scaled, labels):
    """Print cluster statistics and silhouette score."""

    noise_mask = labels == -1
    valid_mask = ~noise_mask

    cluster_labels = labels[valid_mask]
    unique_clusters = np.unique(cluster_labels)

    total_points = len(labels)
    noise_count = int(np.sum(noise_mask))

    print("\n" + "=" * 50)
    print("HDBSCAN CLUSTERING RESULTS")
    print("=" * 50)

    print(f"Total data points: {total_points}")
    print(f"Clusters discovered: {len(unique_clusters)}")
    print(f"Noise points: {noise_count}")
    print(
        f"Noise percentage: "
        f"{noise_count / total_points * 100:.2f}%"
    )

    print("\nCluster sizes:")

    for cluster_id in unique_clusters:
        size = int(np.sum(labels == cluster_id))
        print(f"Cluster {cluster_id}: {size} points")

    # Calculate silhouette score only when valid.
    if (
        len(unique_clusters) >= 2
        and len(cluster_labels) > len(unique_clusters)
    ):
        score = silhouette_score(
            X_scaled[valid_mask],
            cluster_labels,
        )
        print(
            f"\nSilhouette Score "
            f"(excluding noise): {score:.4f}"
        )
    else:
        print(
            "\nSilhouette Score unavailable: "
            "at least two valid clusters are required."
        )

    print("=" * 50)


def visualize_clusters(X, labels):
    """Visualize clusters and noise observations."""

    plt.figure(figsize=(10, 7))

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10")

    for index, label in enumerate(unique_labels):
        mask = labels == label

        if label == -1:
            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                color="black",
                marker="x",
                s=35,
                alpha=0.8,
                label="Noise",
            )
        else:
            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                color=cmap(index % 10),
                s=30,
                alpha=0.75,
                label=f"Cluster {label}",
            )

    plt.title("Day 30: HDBSCAN Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def main():
    """Run the complete HDBSCAN workflow."""

    print("Day 30: HDBSCAN Clustering")
    print("-" * 50)

    # Step 1: Create synthetic data
    X = create_dataset()
    print(f"Dataset shape: {X.shape}")

    # Step 2: Train HDBSCAN
    model, scaler, X_scaled, labels = train_hdbscan(X)

    # Step 3: Evaluate results
    evaluate_clustering(X_scaled, labels)

    # Step 4: Visualize clusters
    visualize_clusters(X, labels)

    print("\nHDBSCAN clustering completed successfully.")


if __name__ == "__main__":
    main()
