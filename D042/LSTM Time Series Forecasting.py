
# Day 42 - Long Short-Term Memory (LSTM) Networks

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 1. Generate time-series dataset
time = np.arange(0, 100, 0.1)

data = (
    np.sin(time)
    + 0.5 * np.sin(2 * time)
    + 0.1 * np.random.randn(len(time))
)

data = data.reshape(-1, 1)

# 2. Normalize dataset
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 3. Create sequences
def create_sequences(dataset, sequence_length=20):

    X, y = [], []

    for i in range(len(dataset) - sequence_length):
        X.append(dataset[i:i + sequence_length])
        y.append(dataset[i + sequence_length])

    return np.array(X), np.array(y)

sequence_length = 20

X, y = create_sequences(scaled_data, sequence_length)

# 4. Train-test split
split = int(len(X) * 0.8)

X_train = torch.FloatTensor(X[:split])
y_train = torch.FloatTensor(y[:split])

X_test = torch.FloatTensor(X[split:])
y_test = torch.FloatTensor(y[split:])

# 5. Define LSTM model
class LSTMModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        output = self.fc(output[:, -1, :])

        return output

# 6. Initialize model
model = LSTMModel()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# 7. Train model
epochs = 30

for epoch in range(epochs):

    model.train()

    predictions = model(X_train)

    loss = criterion(predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Loss: {loss.item():.6f}"
        )

# 8. Make predictions
model.eval()

with torch.no_grad():
    predictions = model(X_test).numpy()

# 9. Reverse normalization
predictions = scaler.inverse_transform(predictions)

actual = scaler.inverse_transform(y_test.numpy())

# 10. Model evaluation
mae = mean_absolute_error(actual, predictions)

rmse = np.sqrt(
    mean_squared_error(actual, predictions)
)

print("\nLSTM MODEL PERFORMANCE")
print("----------------------")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

# 11. Visualize predictions
plt.figure(figsize=(12, 5))

plt.plot(actual, label="Actual Values")
plt.plot(predictions, label="LSTM Predictions")

plt.title("Day 42 - LSTM Time Series Forecasting")
plt.xlabel("Time Steps")
plt.ylabel("Value")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 12. Save trained model
torch.save(model.state_dict(), "lstm_model.pth")

print("\nModel saved successfully!")