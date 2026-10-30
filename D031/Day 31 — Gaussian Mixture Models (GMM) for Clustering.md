# Day 31 — Gaussian Mixture Models (GMM) for Clustering

## Overview:

This project demonstrates **Gaussian Mixture Models (GMM)**, a probabilistic unsupervised-learning algorithm for grouping unlabeled observations. The example uses a synthetic customer dataset with two behavior features: annual spend and visits per month.

Unlike K-Means, GMM estimates a probability distribution for each cluster and can model clusters with different shapes and orientations. Each sample receives a predicted cluster label and a probability of membership in every component.

> **Dataset note:** The data is generated inside the Python script for reproducibility. It is not real customer data, and its results should not be treated as business findings.

## Learning objectives

- Understand probabilistic clustering and Gaussian mixture components.
- Standardize numerical features before fitting the model.
- Compare candidate component counts using Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).
- Interpret cluster labels and membership probabilities.
- Evaluate clusters with the silhouette score and, for this synthetic dataset only, Adjusted Rand Index (ARI).
- Export results and plots for review.

## Project structure

```text
D031/
├── Gaussian_Mixture_Models_Clustering.py
├── README.md
└── outputs/                         # Created when the script runs
    ├── gmm_customer_segments.csv
    ├── gmm_model_selection.csv
    ├── gmm_customer_clusters.png
    └── gmm_aic_bic_comparison.png
```

## Requirements

- Python 3.10 or later recommended
- NumPy
- Pandas
- Matplotlib
- scikit-learn

Install the dependencies from a terminal:

```bash
python -m pip install numpy pandas matplotlib scikit-learn
```

## Run the project

From inside the `D031` directory:

```bash
python Gaussian_Mixture_Models_Clustering.py
```

On Windows, if you are using Git Bash and a virtual environment named `.venv`, activate it first with:

```bash
source .venv/Scripts/activate
```

Then install the dependencies and run the script. Activation is optional if you prefer to use your normal Python installation.

## Workflow

1. Generate three synthetic customer segments with different covariance patterns.
2. Standardize the two input features using `StandardScaler`.
3. Fit GMM candidates with 1–6 components.
4. Compare AIC and BIC and select the candidate with the lowest BIC.
5. Predict cluster labels and membership probabilities.
6. Calculate the silhouette score where it is defined. Calculate ARI only because the generated demo data includes known segment labels.
7. Save the assignments, model-selection table, and plots to `outputs/`.

## Output files

| File | Description |
|---|---|
| `gmm_customer_segments.csv` | Original feature values, predicted cluster, maximum membership probability, and per-cluster probabilities |
| `gmm_model_selection.csv` | AIC, BIC, convergence status, and iteration count for each candidate model |
| `gmm_customer_clusters.png` | Scatter plot of standardized features colored by predicted cluster, with component means |
| `gmm_aic_bic_comparison.png` | AIC/BIC comparison across candidate component counts |

## Key concepts

- **Gaussian mixture:** A probability distribution represented as a weighted combination of Gaussian distributions.
- **Component:** One Gaussian distribution within the mixture. A component is often interpreted as a cluster, though they are not always equivalent in complex data.
- **Soft assignment:** GMM returns membership probabilities for each sample; the displayed cluster label is the component with the highest probability.
- **AIC and BIC:** Model-selection criteria that balance fit and complexity. Lower values are generally preferred when comparing models fitted to the same data. BIC is used for the script's component-count choice.
- **Silhouette score:** Measures how well-separated the assigned clusters are, when the predicted number of clusters makes the metric valid.
- **Adjusted Rand Index:** Compares predicted groups with known labels. Here it is only available because the tutorial data was generated with known segment membership.

## GMM vs. K-Means

| Aspect | K-Means | GMM |
|---|---|---|
| Assignment | Hard assignment | Probabilistic assignment |
| Cluster assumptions | Distance-based, roughly spherical clusters work well | Can represent different elliptical shapes with covariance modeling |
| Output | Cluster label and centroid | Cluster label, component parameters, and membership probabilities |
| Choosing cluster count | Often explored with elbow/silhouette analysis | Can be compared using AIC/BIC, alongside domain judgment |

Neither method is universally best. The right choice depends on the data and the business question.

## Practical interpretation

In a real customer-segmentation project, cluster profiles and membership probabilities could help analysts identify groups for further investigation. Before using the groups for decisions, validate the segments with domain experts, test stability, consider privacy and fairness, and use a representative real dataset.

## Troubleshooting

- **`ModuleNotFoundError`:** Install the listed dependencies in the same Python environment used to run the script.
- **Plot windows do not appear:** The script saves PNG files in `outputs/`; open them from that folder.
- **Different labels/order:** Cluster IDs are arbitrary. Cluster `0` does not inherently mean the lowest-spending group.
- **Different component count:** BIC is a model-selection aid, not a guarantee that the chosen number matches the business's preferred segments.

## Git Bash commands

If the two files are currently in your repository root, use:

```bash
mkdir -p D031
cp Gaussian_Mixture_Models_Clustering.py README.md D031/
git add D031/
git commit -m "Day 31: Gaussian Mixture Models clustering"
git pull --rebase origin main
git push origin main
```

If the files are already inside `D031`, omit the `mkdir` and `cp` lines. If Git reports a conflict during rebase, resolve the reported file(s), then run `git add <file>` and `git rebase --continue`; do not delete files just to bypass a conflict.

## Challenge checkpoint

**Day 31 complete:** Gaussian Mixture Models (GMM) for probabilistic clustering.

**Next topic:** Day 32 — HDBSCAN clustering (density-based clustering with variable-density groups), subject to keeping the challenge sequence distinct.
