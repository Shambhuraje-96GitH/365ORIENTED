# Day 22 — CatBoost Classification

## 365-Day AIML Engineer GitHub Challenge

**Topic:** CatBoost Classification  
**Day:** 22 of 365  
**Domain:** Machine Learning — Ensemble Learning  
**Level:** Intermediate to Advanced

---

## 1. Overview:

CatBoost (Categorical Boosting) is a gradient boosting machine learning library developed by Yandex. It is designed to build powerful predictive models, particularly for structured and tabular datasets.

CatBoost is useful for classification and regression tasks. One of its major advantages is its native support for categorical features, reducing the need for manual categorical encoding.

In this project, we build a CatBoost classification model using the Titanic dataset to predict whether a passenger survived.

## 2. Objectives

- Understand the fundamentals of CatBoost.
- Learn how gradient boosting works.
- Load and explore a real-world dataset.
- Handle missing numerical and categorical values.
- Train a CatBoost classification model.
- Evaluate performance using classification metrics.
- Visualize the confusion matrix and feature importance.
- Save the trained model for future predictions.

## 3. Technologies Used

- **Python** — Programming language
- **CatBoost** — Gradient boosting classification
- **Pandas** — Data manipulation
- **Scikit-learn** — Dataset loading, data splitting, and evaluation metrics
- **Matplotlib** — Data visualization
- **Seaborn** — Visualization library

## 4. Project Structure

```text
D22/
├── CatBoost_Classification.py
├── Day 22 — CatBoost Classification.md
└── catboost_titanic_model.cbm
```

The `.cbm` model file is generated when the Python program finishes successfully.

## 5. Dataset Description

The project uses the Titanic dataset obtained through Scikit-learn's OpenML dataset-fetching utility.

The selected features are:

| Feature | Description |
|---|---|
| `pclass` | Passenger ticket class |
| `sex` | Passenger's recorded sex |
| `age` | Passenger age |
| `sibsp` | Number of siblings or spouses aboard |
| `parch` | Number of parents or children aboard |
| `fare` | Passenger fare |
| `embarked` | Port of embarkation |
| `survived` | Target variable: 0 = Not survived, 1 = Survived |

The target variable is `survived`, making this a binary classification problem.

**Note:** Dataset availability depends on internet access to OpenML during the first run.

## 6. Installation

Install the required Python libraries using:

```bash
pip install catboost scikit-learn pandas matplotlib seaborn
```

Verify that CatBoost is installed:

```bash
python -c "import catboost; print(catboost.__version__)"
```

## 7. Implementation Workflow

The Python implementation follows these steps:

1. Load the Titanic dataset.
2. Select the required features.
3. Remove rows with missing target labels.
4. Separate input features and the target variable.
5. Convert categorical columns to strings and fill missing categorical values.
6. Split the dataset into training and testing sets.
7. Initialize and train the CatBoost classifier.
8. Generate predictions and predicted probabilities.
9. Evaluate classification performance.
10. Plot the confusion matrix.
11. Visualize feature importance.
12. Save the trained model as `catboost_titanic_model.cbm`.

## 8. Model Configuration

The classifier uses the following configuration:

| Parameter | Value | Purpose |
|---|---:|---|
| `iterations` | 500 | Maximum number of boosting iterations |
| `depth` | 6 | Tree depth |
| `learning_rate` | 0.05 | Controls the contribution of each boosting iteration |
| `loss_function` | `Logloss` | Binary classification loss |
| `eval_metric` | `AUC` | Evaluation metric used during training |
| `random_seed` | 42 | Reproducibility |
| `verbose` | 100 | Displays training progress periodically |

The model receives `sex` and `embarked` as categorical features through the `cat_features` parameter.

## 9. Model Evaluation

The program evaluates the model using the following metrics:

### Accuracy

Measures the proportion of predictions that are correct.

### Precision

Measures how many predicted positive cases are actually positive.

### Recall

Measures how many actual positive cases are correctly identified.

### F1-Score

The harmonic mean of precision and recall.

### ROC-AUC

Measures how effectively the model distinguishes between positive and negative classes across different classification thresholds.

### Confusion Matrix

Displays the counts of true positives, true negatives, false positives, and false negatives.

The program also plots feature importance to show which input features contributed most to the model's predictions.

**Important:** Actual evaluation scores depend on the dataset, train/test split, and model configuration. Run the Python file to obtain the measured results; do not assume a fixed accuracy in advance.

## 10. Running the Project

Navigate to the project folder:

```bash
cd ~/Downloads/GitHub365/D22
```

Execute the Python script:

```bash
python CatBoost_Classification.py
```

After execution, the program displays:

- Dataset preview and missing-value information
- Training and testing sample counts
- Accuracy and ROC-AUC
- Classification report
- Confusion matrix visualization
- Feature-importance visualization
- Saved model confirmation

## 11. Saving the Trained Model

The trained model is saved using:

```python
model.save_model("catboost_titanic_model.cbm")
```

A saved model can be loaded later without retraining:

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model("catboost_titanic_model.cbm")
```

For future predictions, supply new input data with the same feature names, compatible data types, and categorical-feature conventions used during training.

## 12. Advantages of CatBoost

- Native handling of categorical features.
- Effective gradient boosting for tabular datasets.
- Supports classification and regression.
- Provides feature-importance information.
- Offers configurable training parameters.
- Supports model persistence for later inference.

## 13. Limitations

- Training can be computationally expensive for some large datasets.
- Hyperparameter tuning may be needed for optimal results.
- Performance depends on data quality and class balance.
- Feature importance does not establish causation.
- A strong score on one train/test split does not guarantee generalization to unseen production data.

## 14. Real-World Applications

CatBoost can be applied to:

- Customer churn prediction
- Fraud detection
- Credit-risk classification
- Customer response prediction
- Sales forecasting with suitable regression targets
- Healthcare classification research
- Structured business-data modelling

These are potential applications; this project specifically demonstrates Titanic passenger-survival classification.

## 15. Interview Questions

**Q1. What is CatBoost?**  
CatBoost is a gradient boosting library that builds ensembles of decision trees and supports categorical features natively.

**Q2. What is gradient boosting?**  
It is an ensemble technique that builds models sequentially, with each new model improving on errors made by the existing ensemble.

**Q3. What is the purpose of `learning_rate`?**  
It controls how strongly each boosting iteration contributes to the final model.

**Q4. What does `iterations` represent?**  
It specifies the maximum number of boosting iterations.

**Q5. Why use ROC-AUC?**  
ROC-AUC evaluates how well a model ranks positive cases above negative cases across classification thresholds.

**Q6. What is the difference between CatBoost and Random Forest?**  
Random Forest generally builds trees independently and combines their predictions, whereas CatBoost builds trees sequentially using gradient boosting.

**Q7. Why save a trained model?**  
Saving allows the model to be reused for inference without repeating the training process.

## 16. Future Improvements

- Compare CatBoost with XGBoost and LightGBM on the same data split.
- Perform cross-validation.
- Tune hyperparameters.
- Investigate class imbalance and threshold selection.
- Track experiments and model metrics.
- Build a prediction API using FastAPI.
- Add automated testing and a reproducible training pipeline.

## 17. Conclusion

In Day 22 of the 365-Day AIML Engineer GitHub Challenge, we implemented CatBoost classification using the Titanic dataset.

The project demonstrates dataset preparation, native categorical-feature handling, model training, classification evaluation, feature-importance visualization, and model saving.

These skills contribute to building reproducible machine learning projects relevant to AI/ML engineering roles.

---

## GitHub Commit

```bash
git add D22/
git commit -m "Day 22: CatBoost Classification"
git pull --rebase origin main
git push origin main
```

**Challenge Progress:** Day 22 / 365 completed after you run the project, review the results, and push both files to GitHub.

**Next planned topic:** Day 23 — AdaBoost Classification.
