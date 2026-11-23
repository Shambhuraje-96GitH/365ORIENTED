
# Day 41 - Recurrent Neural Networks (RNN)

## Objective
Implement a Recurrent Neural Network using PyTorch
to understand sequential data processing and
time-series forecasting.

## Concepts Covered
- Recurrent Neural Networks
- Sequential Data
- Hidden States
- Sliding Window Technique
- Time-Series Forecasting
- PyTorch Model Training
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Technologies Used
- Python
- NumPy
- Matplotlib
- Scikit-learn
- PyTorch

## Project Description

This project implements a basic RNN model to predict
the next value in a sequential time-series dataset.

A synthetic dataset containing multiple sine-wave
patterns and random noise is generated.

The dataset is normalized and transformed into
sequences using a sliding window technique.

An RNN model learns temporal patterns from
the training data and predicts future values.

## Model Architecture

Input Sequence
      |
      v
Recurrent Neural Network
      |
      v
Hidden State
      |
      v
Fully Connected Layer
      |
      v
Next Value Prediction

## Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Installation

Install the required libraries:

pip install numpy matplotlib scikit-learn torch

## Execution

Run the Python file:

python Day41_RNN_TimeSeries_Forecasting.py

## Expected Output

- Training loss after every 10 epochs
- MAE evaluation score
- RMSE evaluation score
- Actual vs Predicted time-series graph

Note: Evaluation values are generated when the
model is executed. They are not predefined.

## Learning Outcome

Understand how RNNs process sequential information,
maintain hidden states, and learn temporal patterns
for time-series forecasting.

## Author
365 Days AIML Challenge - Day 41
