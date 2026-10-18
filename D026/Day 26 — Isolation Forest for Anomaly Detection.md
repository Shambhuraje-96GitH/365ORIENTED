# Day 26 — Isolation Forest for Anomaly Detection

**365-Day AIML Engineer & MNC-Oriented GitHub Challenge**

## Overview

This project demonstrates anomaly detection with the **Isolation Forest** algorithm from scikit-learn. It generates a synthetic two-feature dataset containing normal observations and deliberately injected anomalies, trains an Isolation Forest model, evaluates its predictions, and visualizes the observations flagged as anomalous.

## Learning Objectives:

- Understand anomaly detection and the difference between normal observations and anomalies.
- Learn how Isolation Forest identifies observations that are easier to isolate from the rest of the data.
- Prepare data and split it into training and test sets.
- Evaluate results using precision, recall, F1-score, and a confusion matrix.
- Visualize predictions with Matplotlib.

## Algorithm: Isolation Forest

Isolation Forest is an unsupervised anomaly-detection algorithm. Rather than building a detailed profile of normal observations, it isolates observations through randomly selected features and split values. Anomalies often require fewer splits to isolate.

In scikit-learn, `predict()` returns:
- `1` for an inlier (normal observation).
- `-1` for an outlier (anomaly).

This project maps those outputs to conventional labels:
- `0` = normal
- `1` = anomaly

## Dataset

The script creates a reproducible synthetic dataset using NumPy:

- **1,000 normal observations** sampled around the center of a two-dimensional distribution.
- **60 injected anomalies** sampled across a wider range.
- **2 features:** `Feature_1` and `Feature_2`.
- A fixed random seed is used so the dataset can be reproduced.

The data is split into training and test sets using a stratified split. `StandardScaler` is fitted only on the training features and then applied to the test features to avoid data leakage.

> This is a controlled educational dataset. Its artificial anomaly pattern is simpler than many real-world anomaly-detection problems.

## Technologies Used-

- Python
- NumPy
- pandas
- scikit-learn
- Matplotlib

## Project Structure

```text
D26/
├── Day26_Isolation_Forest_Anomaly_Detection.py
└── Day 26 — Isolation Forest Anomaly Detection.md
```

## Installation

Make sure Python is installed, then install the required packages:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Run the Project

Open a terminal in the `D26` folder and run:

```bash
python Day26_Isolation_Forest_Anomaly_Detection.py
```

The script prints:
- Dataset dimensions and the counts of normal observations and injected anomalies.
- Precision, recall, and F1-score for anomaly predictions.
- A classification report and confusion matrix.
- A sample of observations predicted as anomalies.

It then opens a scatter plot showing predicted normal observations and predicted anomalies.

## Model Configuration

The Isolation Forest model uses:

```python
IsolationForest(
    n_estimators=200,
    contamination=0.06,
    random_state=42
)
```

- `n_estimators=200`: builds 200 isolation trees.
- `contamination=0.06`: sets the expected proportion of outliers used to define the prediction threshold.
- `random_state=42`: makes the model's randomized behavior reproducible.

**Important:** `contamination` is a modeling assumption, not a guarantee that exactly 6% of every test set will be predicted as anomalous. Tune it using domain knowledge and appropriate validation data.

## Evaluation Metrics

- **Precision:** Of the observations flagged as anomalies, the proportion that are actual anomalies.
- **Recall:** Of all actual anomalies, the proportion detected by the model.
- **F1-score:** The harmonic mean of precision and recall.
- **Confusion matrix:** Summarizes actual versus predicted classes.

The script calculates these metrics on the held-out test set. Results can differ if the dataset generation, split, or model parameters change.

## Real-World Applications

- Fraud and suspicious payment detection.
- Network intrusion and unusual traffic monitoring.
- Manufacturing sensor monitoring and predictive maintenance.
- Detecting abnormal system logs or application behavior.
- Quality control and unusual measurements.

## Limitations

- Performance depends on the dataset and the meaning of an anomaly.
- The `contamination` setting affects the decision threshold.
- Anomalies are not always extreme values; some may look normal in individual features.
- Synthetic-data performance does not establish performance on production data.
- Detected observations should be investigated rather than automatically treated as confirmed fraud or failure.

## GitHub Commands

From the repository root:

```bash
git add D26/
git commit -m "Day 26: Isolation Forest Anomaly Detection"
git push origin main
```

## Conclusion

This project provides hands-on practice with unsupervised anomaly detection using Isolation Forest. It covers synthetic data creation, preprocessing, model fitting, evaluation, and visualization—foundational skills for applied machine-learning projects.

**Challenge Progress:** Day 26 of 365 completed after the code and README are added to the repository.
'''
path = Path('/mnt/data/Day 26 — Isolation Forest Anomaly Detection.md')
path.write_text(readme, encoding='utf-8')
print(f"Created: {path}")
