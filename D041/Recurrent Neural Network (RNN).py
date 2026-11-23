
# Day 41: Recurrent Neural Network (RNN)
# Project: Time-Series Forecasting using PyTorch

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 2. Generate Sequential Data
time = np.arange(0, 100, 0.1)

data = (
    np.sin(time)
    + 0.3 * np.sin(3 * time)
    + np.random.normal(0, 0.05, len(time))
)

# 3. Normalize Data
minimum = data.min()
maximum = data.max()

data = (data - minimum) / (maximum - minimum)

# 4. Create Sequences
def create_sequences(dataset, window_size=30):
    X, y = [], []

    for i in range(len(dataset) - window_size):
        X.append(dataset[i:i + window_size])
        y.append(dataset[i + window_size])

    return np.array(X), np.array(y)


X, y = create_sequences(data)

# 5. Chronological Train-Test Split
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# Convert to PyTorch tensors
X_train = torch.tensor(
    X_train, dtype=torch.float32
).unsqueeze(-1)

X_test = torch.tensor(
    X_test, dtype=torch.float32
).unsqueeze(-1)

y_train = torch.tensor(
    y_train, dtype=torch.float32
).unsqueeze(-1)

y_test = torch.tensor(
    y_test, dtype=torch.float32
).unsqueeze(-1)


# 6. Define RNN Model
class RNNModel(nn.Module):

    def __init__(self, input_size=1, hidden_size=32):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        output, hidden = self.rnn(x)

        last_output = output[:, -1, :]

        prediction = self.fc(last_output)

        return prediction


# 7. Initialize Model
model = RNNModel()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# 8. Train Model
epochs = 50

for epoch in range(epochs):

    model.train()

    predictions = model(X_train)

    loss = criterion(predictions, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {loss.item():.6f}"
        )


# 9. Evaluate Model
model.eval()

with torch.no_grad():
    predictions = model(X_test).numpy()

actual = y_test.numpy()

mae = mean_absolute_error(actual, predictions)

rmse = np.sqrt(
    mean_squared_error(actual, predictions)
)

print("\nRNN MODEL EVALUATION")
print("--------------------")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))


# 10. Visualize Predictions
plt.figure(figsize=(12, 5))

plt.plot(
    actual,
    label="Actual Values",
    color="blue"
)

plt.plot(
    predictions,
    label="Predicted Values",
    color="red"
)

plt.title("RNN Time-Series Forecasting")
plt.xlabel("Time Steps")
plt.ylabel("Normalized Value")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
