Day 25 — Stacking Classifier
365-Day AIML Engineer & MNC-Oriented GitHub Challenge
Overview
In this task, we build a Stacking Classifier, an ensemble-learning method that combines several base machine-learning models and learns how to combine their predictions with a final model called a meta-model.
The implementation uses the Breast Cancer Wisconsin dataset built into scikit-learn. It trains and evaluates the ensemble, then prints classification metrics and example predictions.
Important: This is an educational machine-learning exercise, not a medical diagnostic tool. The model's predictions must not be used to make healthcare decisions.

Learning objectives:
By completing this task, you will learn to:
- Explain stacking and the role of a meta-model.
- Combine different classification algorithms in one ensemble.
- Use pipelines to scale features for models that are sensitive to feature scale.
- Use cross-validation to generate training predictions for the meta-model.
- Evaluate a classifier with accuracy, ROC-AUC, a classification report, and a confusion matrix.
What is stacking?
Stacking is an ensemble technique that trains multiple base estimators and uses their predictions as input features for a final estimator (the meta-model). The meta-model learns how to combine the base models' predictions to make the final classification.
The script uses this structure:
Input features
      |
      +--------------------+-------------------+
      |                    |                   |
Logistic Regression   Decision Tree           KNN
      |                    |                   |
      +--------------------+-------------------+
                           |
               Cross-validated predictions
                           |
               Logistic Regression
                 (meta-model)
                           |
                  Final prediction
During training, StackingClassifier uses cross-validated predictions from the base estimators to train the meta-model, helping avoid training the meta-model only on predictions from base models fitted to those same rows.
Models used
Component	Configuration	Purpose
Logistic Regression	StandardScaler() + LogisticRegression(max_iter=2000, random_state=42)	Linear classification baseline; feature scaling is included in its pipeline.
Decision Tree	DecisionTreeClassifier(max_depth=4, random_state=42)	Learns nonlinear decision rules.
K-Nearest Neighbors (KNN)	StandardScaler() + KNeighborsClassifier(n_neighbors=5)	Predicts from nearby training examples; feature scaling is included in its pipeline.
Meta-model	LogisticRegression(max_iter=2000, random_state=42)	Learns how to combine the base-estimator predictions.


The stacking configuration uses cv=5, stack_method="auto", passthrough=False, and n_jobs=-1.
Dataset
- Name: Breast Cancer Wisconsin dataset
- Source: Included with sklearn.datasets.load_breast_cancer; no separate download is needed.
- Samples: 569
- Input features: 30 numeric features
- Classes: malignant and benign
- Target encoding: 0 = malignant, 1 = benign
- Test split: 20% of the dataset
- Split configuration: random_state=42 and stratify=y to preserve class proportions in the train and test sets.
Project files
D25/
├── Day25_Stacking_Classifier.py
└── Day 25 — Stacking Classifier.md
Requirements
- Python 3.9 or later is recommended.
- NumPy
- scikit-learn
- Matplotlib is listed in the challenge's installation command, although this script does not create plots.
Install the dependencies in your terminal:
pip install numpy scikit-learn matplotlib
How to run
1. Open a terminal in the D25 folder.
2. Install the dependencies using the command above.
3. Run the Python script:
python Day25_Stacking_Classifier.py
The script prints dataset information, fits the stacking ensemble, reports its evaluation metrics, displays a confusion matrix, and shows up to ten example predictions with the predicted probability of the benign class.
Evaluation metrics
- Accuracy: The fraction of test examples classified correctly.
- ROC-AUC: Measures how well the model ranks examples from the two classes across decision thresholds; a value closer to 1 indicates stronger separation on this test set.
- Precision: Of the examples predicted as a class, the fraction that truly belong to that class.
- Recall: Of the actual examples in a class, the fraction correctly identified.
- F1-score: The harmonic mean of precision and recall.
- Confusion matrix: Counts actual versus predicted classes. In this script, rows represent actual classes and columns represent predicted classes, ordered as malignant then benign.
Example output
With the script's current configuration in the tested environment, the evaluation included:
Test accuracy: 0.9825 (98.25%)
ROC-AUC:       0.9921

Confusion matrix (rows = actual, columns = predicted):
[[40  2]
 [ 0 72]]
Results can vary with library versions, data handling, or changes to the model configuration. Test-set performance is not a guarantee of performance on new data.
Key takeaways
1. Stacking combines multiple models rather than relying on a single estimator.
2. Base models can learn different patterns, and the meta-model learns how to combine their predictions.
3. Pipelines help ensure feature scaling is applied consistently when the scale-sensitive models are fitted and evaluated.
4. Evaluation should include more than accuracy, particularly when the classes have different consequences.
5. This dataset is for learning and experimentation only; the result is not suitable for clinical use.
Git commands
From the root of your GitHub365 repository, add and commit the Day 25 folder:
git add D25/
git commit -m "Day 25: Stacking Classifier"
git push origin main
If Git reports that the remote contains work you do not have locally, run git pull --rebase origin main and resolve any conflicts before pushing again.
Challenge progress: Day 25 of 365 — Stacking Classifier.
