# Day 24 — Stacking Classifier (Advanced Ensemble Learning)

**365-Day AIML Engineer GitHub Challenge**

## Overview

Stacking is an ensemble-learning technique that combines multiple machine-learning models. The predictions produced by the base models become input features for a final model, commonly called the **meta-model**, which learns how to combine those predictions.

In this project, we train and compare three individual classifiers with a stacking ensemble on scikit-learn's built-in Breast Cancer Wisconsin dataset:

- Logistic Regression
- Random Forest Classifier
- Support Vector Classifier (SVC)
- Stacking Classifier

The project demonstrates model training, evaluation, metric comparison, and visualization in a reproducible classification workflow.

> **Important:** This is an educational machine-learning exercise, not a medical diagnostic tool. Do not use its predictions for clinical decisions.

## Objectives

- Understand the difference between a base estimator and a meta-model.
- Build a stacking ensemble with multiple classifiers.
- Use a pipeline to standardize features for Logistic Regression and SVC.
- Use cross-validation inside `StackingClassifier` to generate out-of-fold predictions for training the final estimator.
- Evaluate models with Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
- Review a classification report and confusion matrix.
- Visualize and compare the evaluated models.

## Dataset

The code loads the **Breast Cancer Wisconsin dataset** directly from `sklearn.datasets` with `load_breast_cancer(as_frame=True)`. No manual download is needed.

The dataset contains numeric features computed from digitized images of breast mass samples. The target represents two classes: malignant and benign. The script uses a stratified 80/20 train-test split to help preserve the class distribution in each set.

## Project Structure

```text
D24/
├── Stacking_Classifier.py
└── README.md
```

## Requirements

- Python 3.9 or later recommended
- scikit-learn
- pandas
- matplotlib

Install the dependencies from a terminal:

```bash
python -m pip install scikit-learn pandas matplotlib
```

## How to Run

1. Open a terminal in the `D24` directory.
2. Install the dependencies using the command above.
3. Run the Python script:

   ```bash
   python Stacking_Classifier.py
   ```

The script trains the individual models and the stacking ensemble, prints a model-performance comparison and a classification report, displays the confusion matrix values, and opens two plots.

## How the Stacking Model Works

1. **Base estimators:** Logistic Regression, Random Forest, and SVC learn from the training data.
2. **Out-of-fold predictions:** With `cv=5`, the stacking classifier obtains cross-validated predictions from the base estimators for training the final estimator. This helps the meta-model learn from predictions made on folds not used to fit the corresponding base model.
3. **Meta-model:** A Logistic Regression model combines the base estimators' predicted probabilities.
4. **Final predictions:** At prediction time, the trained base estimators produce probabilities, which are passed to the meta-model to produce the final class prediction.

The script uses `stack_method="predict_proba"` and a fixed random state for reproducibility. Individual results can still vary with library versions or execution environments.

## Evaluation Metrics

| Metric | Meaning |
|---|---|
| Accuracy | Fraction of all predictions that are correct. |
| Precision | Fraction of positive predictions that are correct. |
| Recall | Fraction of actual positive cases that are identified. |
| F1 Score | Harmonic mean of precision and recall. |
| ROC-AUC | Measures how well predicted probabilities distinguish the two classes across thresholds. |

The program prints a comparison table for all four models and a detailed classification report and confusion matrix for the stacking classifier. It also plots the five metrics for model comparison and a confusion-matrix visualization.

**Interpret the results rather than assuming stacking is always best.** An ensemble may outperform, match, or underperform its individual base models, depending on the dataset, split, and model choices.

## Expected Output

The terminal output includes:

- Dataset dimensions and class distribution
- Number of training and testing samples
- Training progress for each model
- Model metric comparison table
- Stacking classifier classification report
- Confusion matrix
- Model with the highest F1 score in the current run

Two Matplotlib plots should appear: a model metric comparison chart and a confusion matrix for the stacking classifier. Close each plot window to continue if the next plot does not appear immediately.

Exact metric values depend on the installed library versions and the execution environment; use the numbers printed by your own run in any analysis.

## Key Takeaways

- Stacking learns how to combine predictions from different model types.
- Cross-validated, out-of-fold base-model predictions are used to train the meta-model.
- Pipelines keep feature scaling within model training, helping avoid preprocessing leakage between training and test sets.
- Multiple evaluation metrics offer a more complete view than accuracy alone.
- Model selection should be based on validation evidence and the task's requirements, not on an assumption that a more complex ensemble must perform better.

## Git Commands

Run these commands from the **repository root** (`365ORIENTED`), assuming your files are saved in `D24/`:

```bash
git add D24/Stacking_Classifier.py D24/README.md
git commit -m "Day 24: Stacking Classifier"
git push origin main
```

If you are already inside the `D24` directory, use `git add Stacking_Classifier.py README.md` instead.

## Day 24 Complete

**Topic:** Stacking Classifier — Advanced Ensemble Learning  
**Next challenge:** Day 25 — continue with a new AIML topic without repeating a previous task.
