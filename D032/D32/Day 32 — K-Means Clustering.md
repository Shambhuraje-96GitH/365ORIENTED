
# Day 32 — K-Means Clustering

## Project: Customer Segmentation Using K-Means

### Overview
This project demonstrates K-Means, an unsupervised machine learning
algorithm that groups similar data points into clusters.

The project uses synthetic customer data containing annual income
and spending score to demonstrate customer segmentation.

### Learning Objectives
- Understand unsupervised machine learning.
- Implement K-Means using Scikit-learn.
- Apply feature scaling.
- Explore the Elbow Method.
- Evaluate clustering with the Silhouette Score.
- Visualize clusters and their centroids.
- Export clustered data to CSV.

### Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

### Project Structure

```text
D032/
├── KMeans_Clustering.py
├── README.md
└── kmeans_customer_segments.csv  # Generated after execution
```

### Installation

Install Python 3.10 or newer, then run:

```bash
python -m pip install numpy pandas matplotlib scikit-learn
```

### Run the Project

```bash
python KMeans_Clustering.py
```

### Dataset
The project generates a reproducible synthetic dataset with two features:

- Annual Income: customer annual income in thousands.
- Spending Score: a simulated score between 1 and 100.

The data is generated for educational purposes and is not real customer data.

### Methodology

1. Generate the synthetic customer dataset.
2. Inspect the dataset and descriptive statistics.
3. Standardize the input features.
4. Calculate inertia for different cluster counts.
5. Calculate Silhouette Scores for candidate cluster counts.
6. Fit a K-Means model with three clusters.
7. Display cluster profiles and centroids.
8. Visualize the clusters, Elbow Method, and Silhouette Scores.
9. Export the results to a CSV file.

### Evaluation Metrics

**Inertia:** Measures the within-cluster sum of squared distances.
Lower values indicate tighter clusters, but inertia generally decreases
as the number of clusters increases.

**Silhouette Score:** Measures how well-separated and cohesive the
clusters are. Values closer to 1 generally indicate better-defined clusters.

### Output
The script produces:

- Dataset preview and descriptive statistics.
- Number of customers assigned to each cluster.
- Cluster centers and customer profiles.
- Elbow Method plot.
- Silhouette Score plot.
- Customer segmentation scatter plot.
- `kmeans_customer_segments.csv`.

### Business Applications
- Customer segmentation.
- Targeted marketing campaigns.
- Customer behavior analysis.
- Product recommendation strategies.
- Retail and e-commerce analytics.

### Limitations
- The number of clusters must be selected.
- Results can be affected by outliers and feature scaling.
- K-Means favors roughly spherical clusters.
- Synthetic data does not establish real-world business performance.

### Key Takeaway
K-Means provides a practical way to discover groups in unlabeled data.
Feature scaling, cluster selection, and evaluation are important parts
of building a reliable clustering workflow.

### Day 32 Status
Completed: K-Means Clustering
