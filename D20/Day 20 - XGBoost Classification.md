# Day 20 - XGBoost Classification

## 365-Day AIML GitHub Challenge

### Overview

XGBoost (Extreme Gradient Boosting) is a powerful supervised machine learning algorithm based on gradient-boosted decision trees. It combines multiple decision trees sequentially to improve predictive performance.

In this project, an XGBoost classifier is trained on the Breast Cancer Wisconsin dataset to predict whether a tumor is malignant or benign.

### Objectives

- Understand the fundamentals of XGBoost.
- Implement binary classification using Python.
- Train and test an XGBoost model.
- Evaluate model performance using classification metrics.
- Visualize the confusion matrix.
- Identify important features using feature importance scores.

### Technologies Used

- Python
- XGBoost
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

### Dataset

The project uses the Breast Cancer Wisconsin dataset provided by Scikit-learn.

- **Total samples:** 569
- **Features:** 30
- **Problem type:** Binary classification
- **Classes:** Malignant and Benign

Target labels:

- `0` — Malignant
- `1` — Benign

### Project Structure

```text
D20/
├── XGBoost_Classification.py
└── Day 20 - XGBoost Classification.md
```

### Installation

Install the required libraries:

```bash
pip install xgboost scikit-learn pandas matplotlib seaborn
```

### How to Run

Execute the Python file:

```bash
python XGBoost_Classification.py
```

### Implementation Steps

1. Load the Breast Cancer dataset.
2. Separate input features and target labels.
3. Split the dataset into training and testing sets.
4. Initialize the `XGBClassifier`.
5. Train the model using the training dataset.
6. Predict the test-set class labels and probabilities.
7. Evaluate the model using accuracy, ROC-AUC, precision, recall, and F1-score.
8. Visualize the confusion matrix.
9. Plot the ten most important features.

### Model Configuration

The classifier uses the following parameters:

| Parameter | Value |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 3 |
| `learning_rate` | 0.1 |
| `objective` | `binary:logistic` |
| `eval_metric` | `logloss` |
| `random_state` | 42 |

### Evaluation Metrics

- **Accuracy:** Measures the proportion of correct predictions.
- **Precision:** Measures the correctness of positive predictions.
- **Recall:** Measures how many actual positive cases are identified.
- **F1-score:** Balances precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between classes.

### Expected Output

The program displays:

- Dataset information
- Training and testing sample counts
- Model training status
- Accuracy and ROC-AUC scores
- Classification report
- Confusion matrix
- Top ten feature importance visualization

Actual metrics are calculated when the program runs.

### Applications of XGBoost

- Fraud detection
- Credit risk prediction
- Customer churn prediction
- Medical data classification
- Spam detection
- Business analytics

### Key Learnings

- Gradient boosting and ensemble learning
- XGBoost classifier implementation
- Model training and prediction
- Classification performance evaluation
- Confusion matrix interpretation
- Feature importance analysis

### Conclusion

This project demonstrates the practical implementation of XGBoost classification using Python. It develops hands-on experience with ensemble learning, model evaluation, and feature interpretation, which are useful skills for AI/ML engineering roles.

### GitHub Commit

```bash
git add D20/
git commit -m "Day 20: XGBoost Classification"
git push origin main
```

**Challenge Progress:** Day 20 / 365

**Next Topic:** Day 21 - LightGBM Classification
