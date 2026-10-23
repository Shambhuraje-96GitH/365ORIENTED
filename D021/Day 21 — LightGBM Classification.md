# Day 21 — LightGBM Classification

## Overview

This project demonstrates **LightGBM (Light Gradient Boosting Machine)** for binary classification using Python and machine learning libraries. LightGBM is a gradient-boosting framework that uses decision trees to deliver efficient training and strong predictive performance.

This project uses the **Breast Cancer Wisconsin dataset** from Scikit-learn to classify tumors as malignant or benign.

## Objectives:

- Understand the fundamentals of LightGBM classification.
- Train a LightGBM classifier on a real-world dataset.
- Evaluate model performance using classification metrics.
- Visualize the confusion matrix and ROC curve.
- Analyze feature importance.
- Save the trained model for future predictions.

## Technologies Used

- Python
- LightGBM
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Dataset

The project uses the Breast Cancer Wisconsin dataset available through Scikit-learn.

**Dataset details:**
- Features: 30 numerical measurements computed from digitized images of breast mass samples.
- Target classes:
  - `0` — Malignant
  - `1` — Benign
- Task: Binary classification.

The dataset is loaded directly using `sklearn.datasets.load_breast_cancer()`.

## Project Workflow

1. Import the required libraries.
2. Load the Breast Cancer Wisconsin dataset.
3. Prepare the feature matrix and target variable.
4. Split the data into training and testing sets.
5. Initialize the LightGBM classifier.
6. Train the model using the training data.
7. Generate predictions on the test data.
8. Evaluate classification performance.
9. Plot the confusion matrix and ROC curve.
10. Visualize feature importance.
11. Save the trained model using Joblib.

## Evaluation Metrics

The model is evaluated using the following metrics:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of positive predictions that are correct.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between the two classes.
- **Confusion Matrix:** Shows correct and incorrect predictions for each class.

Actual scores depend on the dataset split and model configuration.

## Visualizations

The project generates the following visualizations:

- Confusion matrix heatmap.
- Receiver Operating Characteristic (ROC) curve.
- Feature-importance chart.

These visualizations help explain the model's performance and identify influential features.

## Installation

Install the required dependencies:

```bash
python -m pip install lightgbm scikit-learn pandas numpy matplotlib seaborn joblib
```

## How to Run

Clone the repository or navigate to your existing GitHub challenge directory.

Run the Python script:

```bash
python LightGBM_Classification.py
```

The script trains the model, prints evaluation metrics, displays visualizations, and saves the trained model if model saving is implemented in the Python file.

## Expected Outputs

- Classification evaluation metrics in the terminal.
- Confusion matrix visualization.
- ROC curve visualization.
- Feature-importance visualization.
- Saved model file, such as `lightgbm_model.pkl`, if configured in the script.

## Learning Outcomes

After completing this project, you should be able to:

- Explain how gradient boosting works.
- Build a LightGBM classification pipeline.
- Evaluate a binary classification model.
- Interpret feature importance.
- Save and reuse a trained machine learning model.
- Compare LightGBM with XGBoost and traditional decision-tree models.

## Real-World Applications

LightGBM classification can be applied to:

- Fraud detection.
- Customer churn prediction.
- Credit-risk assessment.
- Medical decision-support research.
- Spam detection.
- Customer behavior prediction.

Medical predictions require appropriate clinical validation and must not be used as a substitute for professional diagnosis.

## Key Takeaway

LightGBM is an efficient gradient-boosting algorithm suitable for many structured-data classification problems. This project provides practical experience in model training, evaluation, visualization, and model persistence.

## Author

**365-Day AIML Engineer GitHub Challenge**

**Day 21 of 365 — LightGBM Classification**

## License

This project is intended for educational and portfolio purposes.
