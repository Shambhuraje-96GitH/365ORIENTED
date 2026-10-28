
"""
Day 33: Explainable AI using SHAP
365-Day AIML GitHub Challenge

Objective:
Train a Random Forest regression model and explain its
predictions using SHAP (SHapley Additive exPlanations).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    # 1. Load a built-in dataset
    dataset = load_diabetes(as_frame=True)
    X = dataset.data
    y = dataset.target

    print("=" * 60)
    print("DAY 33: SHAP-BASED EXPLAINABLE AI")
    print("=" * 60)

    print("\nDataset shape:", X.shape)
    print("\nFeature names:")
    print(list(X.columns))

    # 2. Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 3. Train a regression model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # 4. Evaluate the model
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nMODEL PERFORMANCE")
    print("-" * 40)
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

    # 5. Explain the model using SHAP
    print("\nCalculating SHAP values...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # 6. Global feature importance
    mean_abs_shap = np.abs(
        shap_values.values
    ).mean(axis=0)

    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Mean_Absolute_SHAP": mean_abs_shap
    }).sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    print("\nTOP FEATURES BY SHAP IMPORTANCE")
    print(importance.head(10).to_string(index=False))

    # 7. Plot global feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(
        importance["Feature"][::-1],
        importance["Mean_Absolute_SHAP"][::-1]
    )
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.title("Global Feature Importance using SHAP")
    plt.tight_layout()
    plt.savefig("shap_feature_importance.png", dpi=300)
    plt.show()

    # 8. SHAP summary plot
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 9. Explain an individual prediction
    sample_index = 0
    sample_prediction = model.predict(
        X_test.iloc[[sample_index]]
    )[0]

    print("\nINDIVIDUAL PREDICTION EXPLANATION")
    print("-" * 40)
    print(f"Predicted target value: {sample_prediction:.3f}")
    print(f"Actual target value   : {y_test.iloc[sample_index]:.3f}")

    print("\nFeature contributions for this prediction:")
    local_explanation = pd.DataFrame({
        "Feature": X_test.columns,
        "Feature_Value": X_test.iloc[sample_index].values,
        "SHAP_Value": shap_values.values[sample_index]
    })

    local_explanation["Absolute_SHAP"] = (
        local_explanation["SHAP_Value"].abs()
    )

    local_explanation = local_explanation.sort_values(
        "Absolute_SHAP",
        ascending=False
    )

    print(
        local_explanation[
            ["Feature", "Feature_Value", "SHAP_Value"]
        ].head(10).to_string(index=False)
    )

    # 10. Waterfall plot for one prediction
    shap.plots.waterfall(
        shap_values[sample_index],
        max_display=10,
        show=False
    )
    plt.title("SHAP Explanation for One Prediction")
    plt.tight_layout()
    plt.savefig("shap_waterfall.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nDay 33 completed successfully.")
    print("Generated plots:")
    print("1. shap_feature_importance.png")
    print("2. shap_summary.png")
    print("3. shap_waterfall.png")


if __name__ == "__main__":
    main()
