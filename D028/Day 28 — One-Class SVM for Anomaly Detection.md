Day 28 — One-Class SVM for Anomaly Detection
Objective
Learn how to use One-Class Support Vector Machine (One-Class SVM) to identify unusual observations when training data is expected to contain normal examples and anomaly labels are not available during training.
Why this task matters
Anomaly detection is useful for finding unusual transactions, sensor readings, machine behaviour, network activity, or energy-usage patterns. One-Class SVM learns a boundary around normal observations and flags points outside that boundary as potential anomalies.
Concepts covered
- Unsupervised and semi-supervised anomaly-detection workflows
- StandardScaler and preventing data leakage
- sklearn.svm.OneClassSVM
- RBF kernel, nu, and gamma
- Mapping One-Class SVM output (+1 inlier and -1 outlier) to readable labels
- Precision, recall, F1-score, and a confusion matrix
- Visualizing predicted anomalies and the decision boundary
Files
- One_Class_SVM_Anomaly_Detection.py — complete runnable implementation.
- Day 28 — One-Class SVM for Anomaly Detection.md — task notes and instructions.
Requirements
- Python 3.9 or later recommended
- NumPy
- Pandas
- Matplotlib
- scikit-learn
Install the dependencies:
python -m pip install numpy pandas matplotlib scikit-learn
Run the program
Open a terminal in the folder containing the Python file and run:
python One_Class_SVM_Anomaly_Detection.py
The script creates a reproducible synthetic dataset, trains the model on normal observations only, evaluates predictions against held-out test labels, prints classification metrics and a confusion matrix, and displays plots. No external dataset or download is needed.
Implementation workflow
1. Generate normal training observations and a labeled synthetic test set.
2. Fit StandardScaler using only the normal training observations.
3. Train an RBF-kernel One-Class SVM on the scaled normal observations.
4. Predict on the held-out test set. One-Class SVM returns +1 for inliers and -1 for outliers; the script maps those values to 0 (normal) and 1 (anomaly).
5. Evaluate anomaly predictions using precision, recall, F1-score, and a confusion matrix.
6. Plot actual observations, predicted classes, and the estimated decision boundary.
Important parameters
Parameter	Purpose
kernel="rbf"	Uses a nonlinear radial-basis-function boundary.
nu=0.05	Controls the model's regularization and gives an upper bound on the fraction of training errors, as well as a lower bound on the fraction of support vectors. It is not a guaranteed test-set anomaly percentage.
gamma="scale"	Sets the RBF kernel coefficient based on the training data.


These parameters should be tuned and validated for the application and data distribution; the defaults in this exercise are starting points, not universal best settings.
Expected output
The program prints dataset sizes, sample rows, precision, recall, F1-score, a confusion matrix, and a classification report. It then opens two plots: the held-out data's actual labels and the model's predicted labels with a decision boundary.

Exact metrics depend on the code and random seed. The example uses generated data, so its scores must not be presented as results from a real-world dataset or as evidence of production performance.
Limitations and next steps
- Synthetic two-feature data is a teaching example and may be much easier to separate than real data.
- One-Class SVM can be sensitive to feature scaling, nu, gamma, data volume, and the shape of the normal-data distribution.
- Points flagged as anomalies are candidates for investigation, not automatically confirmed faults or fraud.
- For an extension, compare One-Class SVM with Day 26's Isolation Forest and Day 27's Local Outlier Factor on the same dataset, then discuss precision/recall trade-offs.
Day 28 deliverable
Commit the Python implementation and this separate Markdown README in the D028 folder of the 365-day AIML GitHub challenge.
Git Bash commands
Run these commands from the root of your local challenge repository after putting both files inside D028:
git status
git add D028/One_Class_SVM_Anomaly_Detection.py "D028/Day 28 — One-Class SVM for Anomaly Detection.md"
git commit -m "Day 28: One-Class SVM anomaly detection"
git pull --rebase origin main
git push origin main
If Git reports that there is nothing to commit, check git status and confirm the two files are saved inside D028.