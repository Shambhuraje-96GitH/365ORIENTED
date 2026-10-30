
# Day 30 — HDBSCAN Clustering

## Overview :

This project implements Hierarchical Density-Based Spatial
Clustering of Applications with Noise (HDBSCAN), an
unsupervised machine learning algorithm.

HDBSCAN builds a hierarchy of density-based clusters and
selects stable clusters. It can discover clusters with
different densities and identify observations that do not
belong to any cluster.

## Objectives

- Understand HDBSCAN clustering.
- Learn the important HDBSCAN parameters.
- Discover clusters without specifying their number in advance.
- Identify noise points and potential outliers.
- Visualize clusters using Matplotlib.
- Evaluate clustering using the Silhouette Score.

## Technologies Used

- Python
- NumPy
- Matplotlib
- Scikit-learn

## Project Structure

```text
D030/
├── HDBSCAN_Clustering.py
└── Day 30 — HDBSCAN Clustering.md
```

## Installation

Install the required libraries:

```bash
python -m pip install numpy matplotlib scikit-learn
```

This project uses `sklearn.cluster.HDBSCAN`, available in
Scikit-learn 1.3 and later.

If required, upgrade Scikit-learn:

```bash
python -m pip install --upgrade scikit-learn
```

## How to Run

Execute the Python script:

```bash
python HDBSCAN_Clustering.py
```

The program generates synthetic data, trains an HDBSCAN model,
prints clustering statistics, and displays a visualization.

## Dataset

The dataset is generated using Scikit-learn's `make_blobs`
function and NumPy.

It contains:

- Three groups with different levels of spread.
- Additional randomly generated noise points.
- Two numerical features for visualization.

The dataset is synthetic and does not represent a real-world
measurement dataset.

## Algorithm Workflow

1. Generate synthetic observations.
2. Standardize the numerical features.
3. Initialize the HDBSCAN model.
4. Fit the model and assign cluster labels.
5. Count the discovered clusters and noise observations.
6. Calculate the Silhouette Score when appropriate.
7. Visualize clusters and noise points.

## Important Parameters

### min_cluster_size

Specifies the minimum size of a group that can be considered
a cluster.

### min_samples

Controls how conservatively dense regions are identified.
Larger values generally produce more conservative clustering
and may classify more observations as noise.

### metric

Defines the distance metric used by the algorithm.
This implementation uses Euclidean distance.

### cluster_selection_method

The `eom` method selects stable clusters from the hierarchy.

## Evaluation

### Number of Clusters

Counts the unique clusters, excluding the noise label `-1`.

### Noise Points

Counts observations that were not assigned to a cluster.

### Noise Percentage

Measures the proportion of observations classified as noise.

### Silhouette Score

Measures how well-separated clusters are.

The score ranges from -1 to 1. Higher scores generally indicate
better-separated clusters.

In this implementation, the score excludes observations
labeled as noise and is calculated only when at least two
valid clusters are present.

## Applications

- Customer segmentation
- Spatial data analysis
- Anomaly detection
- Scientific data exploration
- Fraud-pattern investigation
- Discovering clusters with different densities

## Advantages

- Does not require the number of clusters in advance.
- Can discover clusters with different densities.
- Identifies noise and potential outliers.
- Uses hierarchical cluster selection.

## Limitations

- Results depend on parameter settings and feature scaling.
- Distance-based clustering may become less effective in
  high-dimensional datasets.
- Performance depends on the dataset's structure and size.
- Noise points are not necessarily true anomalies.

## Key Learnings

HDBSCAN combines density-based clustering with hierarchical
cluster selection. It is useful when the number of clusters
is unknown and the dataset contains groups with different
densities.

## Challenge Progress

- Day 28: DBSCAN Clustering
- Day 29: OPTICS Clustering
- Day 30: HDBSCAN Clustering

## Next Task

Day 31 — Gaussian Mixture Models (GMM).
