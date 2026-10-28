
"""
Day 32: K-Means Clustering
Project: Customer Segmentation Using K-Means

Skills :
- Data preprocessing and feature scaling
- Elbow Method
- K-Means clustering
- Silhouette Score
- Cluster visualization
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


def create_customer_dataset(n_samples=300, random_state=42):
    """Generate reproducible synthetic customer data."""
    rng = np.random.default_rng(random_state)

    profiles = [
        (120, 25, 30),   # lower income, lower spending
        (220, 80, 85),   # higher income, higher spending
        (180, 55, 35),   # moderate income, moderate spending
    ]

    rows = []

    for count, income_mean, spending_mean in profiles:
        income = rng.normal(income_mean, 12, count)
        spending = rng.normal(spending_mean, 10, count)

        for inc, spend in zip(income, spending):
            rows.append({
                "Annual_Income_k": max(10, inc),
                "Spending_Score": np.clip(spend, 1, 100),
            })

    data = pd.DataFrame(rows).sample(
        n=n_samples, random_state=random_state
    ).reset_index(drop=True)

    return data


def find_optimal_clusters(X, max_k=10):
    """Calculate inertia and silhouette scores for candidate k values."""
    inertias = []
    silhouette_scores = []
    k_values = range(2, max_k + 1)

    for k in range(1, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        labels = model.fit_predict(X)
        inertias.append(model.inertia_)

        if k >= 2:
            silhouette_scores.append(
                silhouette_score(X, labels)
            )

    return list(k_values), inertias, silhouette_scores


def main():
    # 1. Create example dataset
    data = create_customer_dataset()

    print("\nDAY 32: K-MEANS CLUSTERING")
    print("=" * 45)
    print("\nDataset preview:")
    print(data.head())

    print("\nDataset shape:", data.shape)
    print("\nDescriptive statistics:")
    print(data.describe().round(2))

    # 2. Scale features
    features = ["Annual_Income_k", "Spending_Score"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[features])

    # 3. Evaluate possible cluster counts
    k_values, inertias, scores = find_optimal_clusters(
        X_scaled, max_k=10
    )

    # 4. Fit final K-Means model
    # Select k using the elbow plot and silhouette scores.
    # The synthetic data contains three generated customer profiles.
    optimal_k = 3

    model = KMeans(
        n_clusters=optimal_k,
        random_state=42,
        n_init=10
    )

    data["Cluster"] = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, data["Cluster"])

    # Convert cluster centers back to original units
    centers_original = scaler.inverse_transform(model.cluster_centers_)

    print(f"\nSelected number of clusters: {optimal_k}")
    print(f"Silhouette Score: {score:.3f}")
    print("\nCustomers per cluster:")
    print(data["Cluster"].value_counts().sort_index())

    print("\nCluster centers (original units):")
    centers = pd.DataFrame(
        centers_original,
        columns=features
    )
    print(centers.round(2))

    print("\nCluster profiles:")
    print(
        data.groupby("Cluster")[features]
        .mean()
        .round(2)
    )

    # 5. Plot Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, 11),
        inertias,
        marker="o"
    )
    plt.title("Elbow Method for K-Means")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(range(1, 11))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 6. Plot Silhouette Scores
    plt.figure(figsize=(8, 5))
    plt.plot(
        k_values,
        scores,
        marker="o"
    )
    plt.title("Silhouette Score by Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 7. Visualize clusters and centroids
    plt.figure(figsize=(9, 6))

    for cluster_id in sorted(data["Cluster"].unique()):
        subset = data[data["Cluster"] == cluster_id]
        plt.scatter(
            subset["Annual_Income_k"],
            subset["Spending_Score"],
            label=f"Cluster {cluster_id}",
            alpha=0.7
        )

    plt.scatter(
        centers_original[:, 0],
        centers_original[:, 1],
        marker="X",
        s=250,
        edgecolors="black",
        label="Centroids"
    )

    plt.title("Customer Segmentation Using K-Means")
    plt.xlabel("Annual Income (thousands)")
    plt.ylabel("Spending Score (1–100)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

    # 8. Save results
    output_file = "kmeans_customer_segments.csv"
    data.to_csv(output_file, index=False)

    print(f"\nClustered data saved to: {output_file}")
    print("\nDay 32 completed successfully!")


if __name__ == "__main__":
    main()
