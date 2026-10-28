Day 29 — Time Series Forecasting with ARIMA
Objective
Learn how to forecast future values from time-ordered data using the ARIMA model and evaluate predictions on a chronological test period.
Concepts covered
- Time-series data and date-based indexing
- Chronological train-test splitting (without shuffling)
- ARIMA: Autoregressive Integrated Moving Average
- ARIMA order (p, d, q)
- Forecasting future observations and interpreting confidence intervals
- Model evaluation with Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
- Visualizing actual values against forecasts
Project files
- Time_Series_Forecasting_ARIMA.py — complete runnable example
- synthetic_monthly_demand.csv — generated automatically when the Python file runs
- day29_arima_forecast.png — forecast plot generated automatically when the Python file runs
Requirements
Python 3.10 or newer is recommended.

Install the required libraries:
python -m pip install numpy pandas matplotlib scikit-learn statsmodels
Run the project
Run the following command from the D029 folder:
python Time_Series_Forecasting_ARIMA.py
The program generates a reproducible synthetic monthly demand series, uses the first 96 months for training, forecasts the final 24 months, prints MAE and RMSE, saves a CSV file and forecast plot, and displays the plot.
ARIMA order
The example uses ARIMA(2, 0, 2) with a constant and linear trend (trend="ct"):
- p = 2: uses two autoregressive lags.
- d = 0: the series is not differenced; the generated example models its linear trend directly.
- q = 2: uses two moving-average error terms.
These values are set for demonstration. The synthetic data has a linear trend with stationary autocorrelated noise, so this example models the trend directly with trend="ct". For other series, check stationarity and use time-series validation or a carefully designed model-selection process to select suitable parameters.
Evaluation metrics
- MAE: average absolute difference between actual and predicted values.
- RMSE: square root of the average squared prediction error; it penalizes larger errors more heavily than MAE.
The script calculates both metrics from its test-period predictions. The numerical results depend on the specified synthetic data and model and should not be presented as real-world performance.
Important notes
- The sample data is generated inside the script, so no external dataset download is required.
- The train/test split preserves time order. Randomly shuffling a time series would risk leakage from the future into the past.
- Basic ARIMA does not explicitly model recurring seasonal patterns. For seasonal data, investigate SARIMA/SARIMAX and compare it against simple baselines.
- A confident-looking forecast is not proof of accuracy; compare against baselines and validate using data from the real application.
Example applications
- Monthly product-demand forecasting
- Electricity-demand planning
- Sales and inventory forecasting
- Monitoring operational measurements over time
Day 29 completion checklist
- [ ] Install the dependencies.
- [ ] Run the Python script successfully.
- [ ] Review the MAE and RMSE output.
- [ ] Inspect the actual-versus-forecast plot and confidence interval.
- [ ] Explain the meaning of p, d, and q.
- [ ] Commit both the Python file and this Markdown file to GitHub.
GitHub commit
Use the Git Bash commands shared with this task to add, commit, and push the two Day 29 files.