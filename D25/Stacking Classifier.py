from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    """Load data, train a stacking model, and evaluate its performance."""
    # 1. Load a built-in, real-world classification dataset.
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    class_names = dataset.target_names

    print("=" * 65)
    print("DAY 25: STACKING CLASSIFIER")
    print("=" * 65)
    print(f"Dataset: Breast Cancer Wisconsin")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Classes: {', '.join(class_names)}")
    print("Target meaning: 0 = malignant, 1 = benign")

    # 2. Split the dataset while preserving the class distribution.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 3. Define diverse base estimators.
    # Scaling is included inside the pipelines for models sensitive to scale.
    base_estimators = [
        (
            "logistic_regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=42),
            ),
        ),
        (
            "decision_tree",
            DecisionTreeClassifier(max_depth=4, random_state=42),
        ),
        (
            "knn",
            make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=5),
            ),
        ),
    ]

    # 4. Build the stacking ensemble.
    # During training, cross-validated predictions from base estimators are
    # used as features for the final estimator (the meta-model).
    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=42),
        cv=5,
        stack_method="auto",
        n_jobs=-1,
        passthrough=False,
    )

    # 5. Train the model.
    print("\nTraining the stacking classifier...")
    stacking_model.fit(X_train, y_train)

    # 6. Predict labels and class-1 probabilities (benign).
    y_pred = stacking_model.predict(X_test)
    y_probability = stacking_model.predict_proba(X_test)[:, 1]

    # 7. Evaluate the model.
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probability)
    matrix = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 65)
    print("MODEL EVALUATION")
    print("=" * 65)
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"ROC-AUC:       {roc_auc:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(matrix)

    # 8. Show a few predictions, with probabilities for the benign class.
    print("\nSample predictions:")
    for index, (actual, predicted, probability) in enumerate(
        zip(y_test[:10], y_pred[:10], y_probability[:10]), start=1
    ):
        print(
            f"Sample {index:02d} | Actual: {class_names[actual]:9s} | "
            f"Predicted: {class_names[predicted]:9s} | "
            f"P(benign): {probability:.3f}"
        )

    print("\nHow stacking works:")
    print("1. Base models learn patterns from the training data.")
    print("2. Cross-validated base-model predictions form meta-features.")
    print("3. The logistic-regression meta-model learns from those features.")
    print("4. The trained ensemble makes the final prediction.")
    print("\nNote: This is an educational example, not a diagnostic tool.")


if __name__ == "__main__":
    main()
