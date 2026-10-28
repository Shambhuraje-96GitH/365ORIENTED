# Day 33 — Explainable AI Using SHAP

## Overview

This project demonstrates Explainable Artificial Intelligence (XAI) using SHAP (SHapley Additive exPlanations). It explains how individual features influence the predictions made by a machine learning model.

A Random Forest Regressor is trained on the built-in Diabetes dataset from Scikit-learn. SHAP is then used to interpret the model globally and explain individual predictions.

## Objectives

- Understand the importance of model interpretability.
- Train and evaluate a Random Forest regression model.
- Calculate SHAP values for model predictions.
- Identify important features using mean absolute SHAP values.
- Visualize global and local feature contributions.
- Generate feature importance, summary, and waterfall plots.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- SHAP

## Project Structure

```text
D033/
├── SHAP_Explainable_AI.py
├── README.md
├── shap_feature_importance.png
├── shap_summary.png
└── shap_waterfall.png
```

The PNG files are generated when the Python script runs successfully.

## Dataset

The project uses Scikit-learn's built-in Diabetes dataset.

- Features: 10 numerical input variables.
- Target: A quantitative measure of disease progression.
- Dataset loading: `load_diabetes(as_frame=True)`.

No separate dataset download is required.

## Workflow

1. Load the Diabetes dataset.
2. Split the data into training and testing sets.
3. Train a Random Forest Regressor.
4. Evaluate predictions using MAE, RMSE, and R².
5. Calculate SHAP values with `shap.TreeExplainer`.
6. Rank features by mean absolute SHAP value.
7. Generate global and individual prediction explanations.

## Installation

Install the required dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn shap
```

## Execution

Run the Python file:

```bash
python SHAP_Explainable_AI.py
```

The script prints model evaluation metrics, SHAP feature rankings, and an explanation of an individual prediction.

It also saves three visualization files in the current working directory.

## Key Concepts

### SHAP Values

SHAP values estimate how much each feature contributes to a model prediction relative to a baseline prediction.

- Positive SHAP value: Pushes the prediction higher.
- Negative SHAP value: Pushes the prediction lower.
- Larger absolute SHAP value: Indicates a larger contribution for that observation.

### Global Explainability

Mean absolute SHAP values summarize the magnitude of feature contributions across the evaluated dataset.

### Local Explainability

A waterfall plot explains how feature contributions move an individual prediction away from the model's baseline.

## Evaluation Metrics

- **MAE:** Mean Absolute Error.
- **RMSE:** Root Mean Squared Error.
- **R²:** Coefficient of determination.

Actual metric values should be recorded after running the code; they are not assumed in advance.

## Expected Learning Outcomes

After completing this project, you should be able to:

- Explain the difference between global and local interpretability.
- Use SHAP with tree-based machine learning models.
- Interpret feature contribution plots.
- Communicate model behavior to technical and non-technical stakeholders.
- Recognize the limitations of feature attribution.

## Limitations

SHAP explanations describe the fitted model's behavior; they do not establish causation. Correlated features can complicate attribution, and explanations depend on the model, data, and explainer configuration.

The Diabetes dataset is for educational practice and is not intended for clinical decision-making.

## Industry Relevance

Explainability is useful in machine learning monitoring, model debugging, risk analysis, and applications where stakeholders need to understand model outputs.

## Conclusion

This project introduces an interpretable machine learning workflow by combining Random Forest regression with SHAP explanations. It provides practical experience in explaining both overall model behavior and individual predictions.
