"""Day 29 - Time Series Forecasting with ARIMA

This standalone example creates a reproducible synthetic monthly demand series,
trains an ARIMA model, evaluates a chronological holdout period, and plots the
forecast with confidence intervals.

Install dependencies:
    python -m pip install numpy pandas matplotlib scikit-learn statsmodels

Note: The data is synthetic for learning purposes; metrics are not real-world
business or electricity-demand results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


RANDOM_SEED = 42
TRAIN_MONTHS = 96  # First 8 years
FORECAST_MONTHS = 24  # Next 2 years
ARIMA_ORDER = (2, 0, 2)


def create_synthetic_time_series() -> pd.Series:
    """Create a reproducible monthly demand series with trend and autocorrelation."""
    rng = np.random.default_rng(RANDOM_SEED)
    periods = TRAIN_MONTHS + FORECAST_MONTHS
    dates = pd.date_range(start="2016-01-01", periods=periods, freq="MS")

    # Autocorrelated noise makes the example more realistic than pure random noise.
    errors = np.zeros(periods, dtype=float)
    for i in range(1, periods):
        errors[i] = 0.65 * errors[i - 1] + rng.normal(loc=0.0, scale=2.5)

    trend = 150 + 0.75 * np.arange(periods)
    demand = trend + errors
    return pd.Series(demand, index=dates, name="monthly_demand")


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    series = create_synthetic_time_series()

    # Save the generated dataset so the input data can be inspected separately.
    dataset_path = output_dir / "synthetic_monthly_demand.csv"
    series.rename_axis("date").to_frame().to_csv(dataset_path)

    # Time-series split: never shuffle chronological observations.
    train = series.iloc[:TRAIN_MONTHS]
    test = series.iloc[TRAIN_MONTHS:]

    print("=" * 64)
    print("DAY 29: TIME SERIES FORECASTING WITH ARIMA")
    print("=" * 64)
    print(f"Total observations : {len(series)}")
    print(f"Training months    : {len(train)}")
    print(f"Testing months     : {len(test)}")
    print(f"ARIMA order        : {ARIMA_ORDER}")
    print("\nTraining the model...")

    model = ARIMA(train, order=ARIMA_ORDER, trend="ct")
    fitted_model = model.fit()

    forecast_result = fitted_model.get_forecast(steps=len(test))
    forecast = forecast_result.predicted_mean
    confidence_interval = forecast_result.conf_int(alpha=0.05)

    # Align predictions with the test dates and calculate holdout metrics.
    forecast = pd.Series(np.asarray(forecast), index=test.index, name="forecast")
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))

    print("\nEvaluation on the held-out test period")
    print("-" * 40)
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print("\nFirst five predictions:")
    results = pd.DataFrame(
        {
            "actual": test,
            "forecast": forecast,
        }
    )
    print(results.head().round(2).to_string())

    # Plot training observations, actual test values, forecast, and 95% interval.
    lower = np.asarray(confidence_interval.iloc[:, 0])
    upper = np.asarray(confidence_interval.iloc[:, 1])

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label="Training data", linewidth=1.8)
    plt.plot(test.index, test, label="Actual test data", linewidth=2)
    plt.plot(forecast.index, forecast, label="ARIMA forecast", linestyle="--", linewidth=2)
    plt.fill_between(
        test.index,
        lower,
        upper,
        alpha=0.2,
        label="95% confidence interval",
    )
    plt.axvline(test.index[0], color="gray", linestyle=":", label="Train/test split")
    plt.title("Monthly Demand Forecast Using ARIMA(2, 0, 2)")
    plt.xlabel("Month")
    plt.ylabel("Demand (synthetic units)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    plot_path = output_dir / "day29_arima_forecast.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nDataset saved to : {dataset_path.name}")
    print(f"Forecast plot saved to : {plot_path.name}")
    print("\nReminder: This uses synthetic data for practice, not a real dataset.")
    plt.show()


if __name__ == "__main__":
    main()
