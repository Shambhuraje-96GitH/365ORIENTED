
# Day 42 - Long Short-Term Memory (LSTM) Networks

## Overview

This project demonstrates Long Short-Term Memory (LSTM)
networks using PyTorch.

LSTM is an advanced type of Recurrent Neural Network
designed to learn long-term dependencies in sequential data.

## Objectives

- Understand LSTM architecture.
- Learn Forget, Input and Output Gates.
- Understand hidden state and cell state.
- Implement LSTM using PyTorch.
- Perform time-series forecasting.
- Evaluate model performance.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Dataset

A synthetic time-series dataset is generated using
sine waves and random noise.

The dataset is normalized using MinMaxScaler.

## Implementation Steps

1. Generate time-series data.
2. Normalize the dataset.
3. Create sequential input windows.
4. Split data into training and testing sets.
5. Build an LSTM neural network.
6. Train the model using Adam optimizer.
7. Predict future values.
8. Evaluate predictions using MAE and RMSE.
9. Visualize actual and predicted values.
10. Save the trained model.

## LSTM Architecture

Input Layer
    |
LSTM Layer (64 Hidden Units)
    |
LSTM Layer
    |
Fully Connected Layer
    |
Predicted Output

## Evaluation Metrics

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error

## How to Run

Install dependencies:

pip install numpy torch matplotlib scikit-learn

Execute:

python Day42_LSTM_Networks.py

## Output

- Training loss
- MAE and RMSE scores
- Actual vs predicted graph
- Saved LSTM model

## Conclusion

This project demonstrates how LSTM networks
learn sequential patterns and capture long-term
dependencies for time-series forecasting.

## Author

AIML 365 Days GitHub Challenge
Day 42 of 365