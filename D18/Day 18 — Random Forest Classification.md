# Day 18 — Random Forest Classification

## 365-Day AIML Engineer GitHub Challenge

**Day:** 18
**Topic:** Random Forest Classification
**Domain:** Machine Learning — Supervised Learning
**Language:** Python
**Difficulty:** Intermediate

---

## 1. Objective

The objective of this project is to implement a Random Forest Classification model using Python and Scikit-learn.

This project demonstrates how to train an ensemble machine learning model, predict classes, evaluate classification performance, and identify important features.

## 2. What Is Random Forest?

Random Forest is a supervised machine learning algorithm that combines multiple decision trees to make predictions.

For classification tasks, each decision tree predicts a class, and the forest combines their predictions through voting.

### How It Works

1. Select random samples from the training dataset.
2. Build multiple decision trees using randomized subsets of data and features.
3. Obtain predictions from individual trees.
4. Combine the tree predictions using majority voting.
5. Return the final predicted class.

### Advantages

* Often provides strong classification performance.
* Reduces overfitting compared with an individual, fully grown decision tree in many situations.
* Can model nonlinear relationships.
* Supports feature importance analysis.
* Handles datasets containing many numerical features.

### Limitations

* Can require more memory and computation than a single decision tree.
* May be less interpretable than an individual decision tree.
* Feature importance scores can be misleading when features have different characteristics or are highly correlated.
* Performance depends on dataset characteristics and model configuration.

## 3. Dataset Description

This project uses Scikit-learn's built-in **Breast Cancer Wisconsin dataset**.

The dataset contains numerical measurements derived from digitized images of breast masses. The task is to classify observations as benign or malignant.

| Property       | Description                   |
| -------------- | ----------------------------- |
| Dataset        | Breast Cancer Wisconsin       |
| Total samples  | 569                           |
| Input features | 30                            |
| Target classes | 2                             |
| Missing values | None in the original dataset  |
| Problem type   | Binary classification         |
| Source         | Scikit-learn built-in dataset |

**Important:** This is an educational dataset. The model is not intended for real-world medical diagnosis or clinical decision-making.

## 4. Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn

### Libraries

| Library                  | Purpose                             |
| ------------------------ | ----------------------------------- |
| `pandas`                 | Organizing feature data             |
| `matplotlib`             | Visualizing results                 |
| `sklearn.datasets`       | Loading the dataset                 |
| `train_test_split`       | Splitting training and testing data |
| `RandomForestClassifier` | Training the model                  |
| `sklearn.metrics`        | Evaluating model performance        |

## 5. Project Structure

```text
D18/
├── Random_Forest_Classification.py
└── Day 18 - Random Forest Classification.md
```

## 6. Installation

Install the required dependencies:

```bash
pip install pandas matplotlib scikit-learn
```

## 7. How to Run the Project

Navigate to the `D18` directory and run:

```bash
python Random_Forest_Classification.py
```

The program will:

1. Load the dataset.
2. Display the dataset dimensions and sample records.
3. Split the dataset into training and testing sets.
4. Train a Random Forest classifier.
5. Calculate classification metrics.
6. Display the confusion matrix.
7. Show the ten most important features.
8. Predict a test sample and display its class probabilities.

## 8. Model Configuration

The model uses the following configuration:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)
```

| Parameter                 | Explanation                                                |
| ------------------------- | ---------------------------------------------------------- |
| `n_estimators=100`        | Builds 100 decision trees                                  |
| `max_depth=None`          | Allows trees to grow until other stopping conditions apply |
| `random_state=42`         | Makes the randomized training process reproducible         |
| `class_weight="balanced"` | Adjusts class weights inversely to their frequencies       |

The dataset is split using an 80:20 training/testing ratio. Stratification helps preserve the target class proportions in both sets.

## 9. Evaluation Metrics

### Accuracy

The proportion of all predictions that are correct.

**Formula:**

Accuracy = Correct Predictions / Total Predictions

### Precision

Measures how many observations predicted as a particular class actually belong to that class.

### Recall

Measures how many actual observations of a class are correctly identified.

### F1-Score

The harmonic mean of precision and recall.

F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

### Confusion Matrix

The confusion matrix summarizes correct and incorrect predictions for each class.

It helps identify which classes the model predicts correctly and where it makes mistakes.

**Note:** For this dataset, pay particular attention to recall for the malignant class because false negatives are important in a medical screening context.

## 10. Feature Importance

Random Forest provides feature importance scores through:

```python
model.feature_importances_
```

The program displays the ten highest-ranked features in a horizontal bar chart.

These scores indicate how much features contribute to reductions in impurity across the fitted trees. They do not establish causation or guarantee that a feature is medically meaningful.

## 11. Expected Output

The terminal displays:

* Dataset dimensions and target classes
* Training and testing sample counts
* Model training confirmation
* Accuracy and accuracy percentage
* Classification report
* Confusion matrix
* Ten most important features
* Predicted class and actual class for a test sample
* Predicted probabilities for both classes

Two visualizations are generated:

1. Random Forest Confusion Matrix
2. Top 10 Features — Random Forest

**Note:** Exact metric values depend on the dataset, split, and model configuration. Run the Python file to obtain the actual results rather than relying on assumed accuracy values.

## 12. Practical Applications

Random Forest classification can be applied to:

* Fraud detection
* Customer churn prediction
* Spam classification
* Customer segmentation support
* Quality control
* Risk classification
* Predictive maintenance

Applications in healthcare require appropriate clinical validation, privacy safeguards, and expert oversight.

## 13. Interview Questions

**Q1. What is Random Forest?**

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve predictive performance and robustness.

**Q2. What is ensemble learning?**

Ensemble learning combines predictions from multiple models to produce a final prediction.

**Q3. What is the difference between a Decision Tree and Random Forest?**

A Decision Tree uses one tree to make predictions, whereas Random Forest combines predictions from many trees.

**Q4. What does `n_estimators` mean?**

It specifies the number of decision trees in the forest.

**Q5. Why do we use `train_test_split()`?**

It separates data used to train the model from data used to evaluate its performance on unseen observations.

**Q6. Why is `random_state` used?**

It makes randomized operations reproducible when the data and software environment remain consistent.

**Q7. What is `class_weight="balanced"`?**

It assigns class weights inversely proportional to class frequencies, which can help when classes are imbalanced.

**Q8. What is the difference between precision and recall?**

Precision measures the correctness of positive predictions, while recall measures how many actual positive observations are identified.

**Q9. What is feature importance?**

Feature importance estimates how much each feature contributes to the fitted model according to a particular importance measure.

**Q10. Does Random Forest always outperform a Decision Tree?**

No. Performance depends on the dataset, feature quality, model settings, and evaluation method.

## 14. Learning Outcomes

After completing Day 18, I can:

* Explain Random Forest and ensemble learning.
* Train a classification model with Scikit-learn.
* Split data into training and testing sets.
* Evaluate classification models using multiple metrics.
* Interpret a confusion matrix.
* Visualize feature importance.
* Generate class predictions and probabilities.
* Organize a reproducible machine learning project for GitHub.

## 15. GitHub Commit

From the repository root, execute:

```bash
git add D18/
git commit -m "Day 18: Random Forest Classification"
git pull --rebase origin main
git push origin main
```

If Git reports a rebase conflict, resolve it before pushing.

---

## Conclusion

Day 18 introduced Random Forest Classification through a practical Scikit-learn implementation. The project covers model training, evaluation, confusion matrix visualization, feature importance, and sample predictions.

**Challenge Progress:** Day 18 of 365 completed after the code is executed, reviewed, and committed successfully.

**Next Step:** Day 19 — Gradient Boosting Classification.
