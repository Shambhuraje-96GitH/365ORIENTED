# Day 27 — Local Outlier Factor (LOF) for Anomaly Detection

## 📌 Project Overview

This project implements **Local Outlier Factor (LOF)**, an unsupervised machine learning algorithm used to identify anomalies or unusual observations in datasets.

LOF detects anomalies by comparing the local density of each observation with the local densities of its neighboring observations. Points with significantly lower local density than their neighbors are considered potential anomalies.

This project is part of my **365-Day AIML Engineer GitHub Challenge**, designed to develop practical Artificial Intelligence and Machine Learning skills for industry-oriented roles.

## 🎯 Objectives

- Understand the working principle of Local Outlier Factor.
- Generate a synthetic dataset containing normal observations and anomalies.
- Apply feature scaling using `StandardScaler`.
- Train an LOF anomaly detection model.
- Evaluate anomaly detection performance using classification metrics.
- Visualize predicted anomalies and actual observations.
- Save visualizations for documentation and analysis.

## 🧠 Algorithm: Local Outlier Factor

Local Outlier Factor is a density-based anomaly detection algorithm introduced to identify observations that are isolated relative to their local neighborhoods.

### How It Works

1. **Find nearest neighbors:** Identify the nearest neighboring observations for each data point.
2. **Calculate local reachability density:** Estimate the local density around each observation.
3. **Calculate the LOF score:** Compare an observation's density with the densities of its neighbors.
4. **Identify anomalies:** Observations with substantially lower local density are more likely to be anomalous.

### Key Parameters

| Parameter | Description |
|---|---|
| `n_neighbors=20` | Number of neighbors considered when estimating local density. |
| `contamination` | Expected proportion of anomalies in the dataset. |
| `metric="minkowski"` | Distance metric used to calculate neighborhood distances. |

In scikit-learn, `fit_predict()` returns `1` for inliers and `-1` for outliers.

## 📊 Dataset Description

The dataset is generated using NumPy and scikit-learn's `make_blobs()` function.

| Property | Value |
|---|---|
| Normal observations | 500 |
| Injected anomalies | 30 |
| Total observations | 530 |
| Features | 2 numerical features |
| Random state | 42 |
| Learning approach | Unsupervised anomaly detection |

The normal observations are generated as two clusters, while additional observations are sampled from a wider numerical range to introduce potential anomalies.

The actual anomaly labels are retained only for evaluating the model's predictions.

## 🛠️ Technologies Used

- **Python** — Programming language
- **NumPy** — Numerical computations and data generation
- **Matplotlib** — Data visualization
- **Scikit-learn** — Feature scaling, LOF model, and evaluation metrics

## 📁 Project Structure

```text
D27/
│
├── Local_Outlier_Factor_Anomaly_Detection.py
├── Day 27 — Local Outlier Factor Anomaly Detection.md
├── lof_anomaly_detection.png
└── lof_ground_truth.png
```

The two PNG files are generated automatically when the Python script runs successfully.

## ⚙️ Installation

Install the required Python libraries:

```bash
python -m pip install numpy matplotlib scikit-learn
```

## ▶️ How to Run

1. Open the terminal inside the `D27` directory.
2. Install the required dependencies.
3. Execute the Python script:

```bash
python Local_Outlier_Factor_Anomaly_Detection.py
```

The program prints anomaly detection metrics, a confusion matrix, a classification report, and a detection summary.

It also generates and saves the visualization images.

## 📈 Model Evaluation

The project evaluates the detected anomalies using the following metrics:

- **Accuracy:** Proportion of all observations classified correctly.
- **Precision:** Proportion of predicted anomalies that are actual anomalies.
- **Recall:** Proportion of actual anomalies detected by the model.
- **F1-score:** Harmonic mean of anomaly precision and recall.
- **Confusion matrix:** Shows correct and incorrect predictions for normal observations and anomalies.

The script also displays the indices of the observations with the lowest LOF scores.

Actual metric values depend on the generated data and model predictions. They are printed when the script executes.

## 📉 Visualizations

### 1. Predicted Anomalies

The `lof_anomaly_detection.png` image displays observations classified as normal and those identified as anomalies.

- Blue points represent predicted normal observations.
- Red crosses represent predicted anomalies.

### 2. Ground Truth

The `lof_ground_truth.png` image shows the original labels used to evaluate model performance.

Comparing the two visualizations helps assess how well LOF identifies the injected anomalies.

## 💡 Key Learnings

- Understanding density-based anomaly detection.
- Applying feature scaling before distance-based modeling.
- Configuring LOF using neighborhood size and contamination.
- Interpreting anomaly labels and LOF scores.
- Evaluating an anomaly detector using precision, recall, F1-score, and a confusion matrix.
- Visualizing and saving machine learning results.

## 🏢 Real-World Applications

Local Outlier Factor can be useful in:

- **Fraud detection:** Identifying unusual financial transactions.
- **Cybersecurity:** Detecting suspicious network activity.
- **Manufacturing:** Finding abnormal sensor measurements.
- **Data quality:** Identifying unusual records in datasets.
- **System monitoring:** Detecting unexpected operational behavior.

These applications require appropriate real-world datasets, validation, and domain-specific threshold tuning.

## ⚠️ Limitations

- LOF depends on the choice of neighborhood size.
- Feature scaling can significantly affect distance calculations.
- Anomalies may be difficult to identify when normal data has varying local densities.
- The `contamination` parameter influences the classification threshold.
- In its default novelty-detection setting, LOF is designed to identify outliers in the training data; `fit_predict()` is used here for training-set detection.

## 🏁 Conclusion

This project demonstrates how Local Outlier Factor can identify potentially anomalous observations using local density comparisons.

It combines synthetic data generation, feature scaling, unsupervised anomaly detection, performance evaluation, and visualization into a practical Python project.

**Day 27 completed — 365-Day AIML Engineer GitHub Challenge.**

## 🔗 GitHub Commit

```bash
git add D27
git commit -m "Day 27: Local Outlier Factor Anomaly Detection"
git push origin main
```

---

**Author:** AIML Engineer Challenge Participant

**Challenge:** 365 Days of AI and Machine Learning

**Day:** 27 of 365

**Topic:** Local Outlier Factor (LOF) for Anomaly Detection
