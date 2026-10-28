"""
Day 31 - Gaussian Mixture Models (GMM) for Clustering

Practical project: Customer Segmentation using probabilistic clustering.

The script creates a reproducible synthetic customer dataset, scales the
features, selects a component count using BIC, evaluates the resulting
clusters, and saves visualizations plus a clustered CSV file.

Note: The generated dataset is synthetic and is for learning/demo purposes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
SAMPLES_PER_SEGMENT = 200
MIN_COMPONENTS = 1
MAX_COMPONENTS = 6

FEATURE_NAMES = ["Annual_Spend_Thousands", "Visits_Per_Month"]


def create_synthetic_customer_data(
    samples_per_segment: int = SAMPLES_PER_SEGMENT,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Create three synthetic customer groups with different covariances.

    Returns:
        A DataFrame containing two customer-behavior features and an array of
        hidden segment labels used only to evaluate this synthetic example.
    """
    rng = np.random.default_rng(random_state)

    segment_parameters = [
        # Mean: annual spend (thousands), visits per month; covariance matrix.
        (np.array([25.0, 3.0]), np.array([[36.0, 5.0], [5.0, 1.8]])),
        (np.array([60.0, 9.0]), np.array([[100.0, 12.0], [12.0, 4.0]])),
        (np.array([90.0, 5.0]), np.array([[64.0, -7.0], [-7.0, 2.5]])),
    ]

    data_parts = []
    labels = []
    for segment_id, (mean, covariance) in enumerate(segment_parameters):
        segment_data = rng.multivariate_normal(
            mean=mean,
            cov=covariance,
            size=samples_per_segment,
        )
        data_parts.append(segment_data)
        labels.extend([segment_id] * samples_per_segment)

    data = np.vstack(data_parts)
    customer_df = pd.DataFrame(data, columns=FEATURE_NAMES)
    customer_df = customer_df.clip(lower=0)
    true_labels = np.asarray(labels, dtype=int)
    return customer_df, true_labels


def select_gmm_by_bic(
    scaled_data: np.ndarray,
    min_components: int = MIN_COMPONENTS,
    max_components: int = MAX_COMPONENTS,
    random_state: int = RANDOM_STATE,
) -> tuple[GaussianMixture, pd.DataFrame]:
    """Fit candidate GMMs and return the model with the lowest BIC."""
    if min_components < 1:
        raise ValueError("min_components must be at least 1.")
    if max_components < min_components:
        raise ValueError("max_components must be >= min_components.")
    if max_components > len(scaled_data):
        raise ValueError("max_components cannot exceed the number of samples.")

    fitted_models = {}
    results = []

    for n_components in range(min_components, max_components + 1):
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            n_init=5,
            reg_covar=1e-6,
            random_state=random_state,
        )
        model.fit(scaled_data)
        fitted_models[n_components] = model
        results.append(
            {
                "n_components": n_components,
                "AIC": model.aic(scaled_data),
                "BIC": model.bic(scaled_data),
                "converged": bool(model.converged_),
                "iterations": int(model.n_iter_),
            }
        )

    results_df = pd.DataFrame(results).sort_values("n_components")
    best_component_count = int(results_df.loc[results_df["BIC"].idxmin(), "n_components"])
    best_model = fitted_models[best_component_count]
    return best_model, results_df


def save_component_selection_plot(results_df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot AIC and BIC against the number of mixture components."""
    output_path = output_dir / "gmm_aic_bic_comparison.png"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(results_df["n_components"], results_df["AIC"], marker="o", label="AIC")
    ax.plot(results_df["n_components"], results_df["BIC"], marker="s", label="BIC")
    ax.set_title("Selecting the Number of GMM Components")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Information criterion (lower is better)")
    ax.set_xticks(results_df["n_components"])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_cluster_plot(
    scaled_data: np.ndarray,
    cluster_labels: np.ndarray,
    model: GaussianMixture,
    output_dir: Path,
) -> Path:
    """Save a scatter plot of scaled feature values and GMM means."""
    output_path = output_dir / "gmm_customer_clusters.png"
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        scaled_data[:, 0],
        scaled_data[:, 1],
        c=cluster_labels,
        cmap="viridis",
        alpha=0.7,
        s=28,
        edgecolors="none",
    )
    ax.scatter(
        model.means_[:, 0],
        model.means_[:, 1],
        marker="X",
        s=220,
        c="red",
        edgecolors="black",
        linewidths=0.8,
        label="GMM component means",
    )
    ax.set_title("Customer Segments Identified by GMM")
    ax.set_xlabel("Annual spend (standardized)")
    ax.set_ylabel("Monthly visits (standardized)")
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="Predicted cluster label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    """Run the end-to-end GMM clustering demonstration."""
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("DAY 31: GAUSSIAN MIXTURE MODELS (GMM) FOR CLUSTERING")
    print("=" * 68)

    customer_df, true_labels = create_synthetic_customer_data()
    print(f"\nDataset shape: {customer_df.shape}")
    print("\nFirst five rows:")
    print(customer_df.head().round(2).to_string(index=False))
    print("\nFeature summary:")
    print(customer_df.describe().round(2).to_string())

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(customer_df[FEATURE_NAMES])

    best_model, selection_df = select_gmm_by_bic(scaled_data)
    selected_components = int(best_model.n_components)
    cluster_labels = best_model.predict(scaled_data)
    probabilities = best_model.predict_proba(scaled_data)
    confidence = probabilities.max(axis=1)

    print("\nModel comparison (lower AIC/BIC is generally preferred):")
    print(selection_df.round(2).to_string(index=False))
    print(f"\nNumber of components selected by BIC: {selected_components}")
    print(f"Model converged: {best_model.converged_}")
    print(f"Model iterations: {best_model.n_iter_}")

    unique_clusters = np.unique(cluster_labels)
    if 1 < len(unique_clusters) < len(customer_df):
        silhouette = silhouette_score(scaled_data, cluster_labels)
        print(f"Silhouette score: {silhouette:.3f}")
    else:
        print("Silhouette score: not defined for the predicted cluster count.")

    # Ground-truth labels exist only because this is a synthetic teaching dataset.
    ari = adjusted_rand_score(true_labels, cluster_labels)
    print(f"Adjusted Rand Index (synthetic data only): {ari:.3f}")

    print("\nNumber of customers in each predicted cluster:")
    cluster_counts = pd.Series(cluster_labels, name="Cluster").value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} customers")

    results_df = customer_df.copy()
    results_df["Cluster"] = cluster_labels
    results_df["Membership_Confidence"] = confidence
    for component_id in range(selected_components):
        results_df[f"Probability_Cluster_{component_id}"] = probabilities[:, component_id]

    csv_path = output_dir / "gmm_customer_segments.csv"
    results_df.to_csv(csv_path, index=False)
    selection_csv_path = output_dir / "gmm_model_selection.csv"
    selection_df.to_csv(selection_csv_path, index=False)
    cluster_plot_path = save_cluster_plot(scaled_data, cluster_labels, best_model, output_dir)
    selection_plot_path = save_component_selection_plot(selection_df, output_dir)

    print("\nCluster profiles (means in original feature units):")
    print(results_df.groupby("Cluster")[FEATURE_NAMES].mean().round(2).to_string())

    print("\nFiles saved:")
    for path in (csv_path, selection_csv_path, cluster_plot_path, selection_plot_path):
        print(f"  - {path}")

    print("\nDay 31 completed successfully.")
    print("Reminder: synthetic results are for demonstration, not business conclusions.")


if __name__ == "__main__":
    main()
