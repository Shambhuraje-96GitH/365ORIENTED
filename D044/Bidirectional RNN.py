
# Day 44: Bidirectional RNN
# Electricity Consumption Forecasting

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Bidirectional

# 1. Generate electricity consumption data
np.random.seed(42)

time = np.arange(300)

data = (
    100
    + 20 * np.sin(time * 0.1)
    + 10 * np.sin(time * 0.03)
    + np.random.normal(0, 3, 300)
)

data = data.reshape(-1, 1)

# 2. Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 3. Create sequences
def create_sequences(data, window=10):

    X, y = [], []

    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])

    return np.array(X), np.array(y)


X, y = create_sequences(scaled_data)

# 4. Split training and testing data
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Build Bidirectional RNN model
model = Sequential([
    Bidirectional(
        SimpleRNN(32, activation="tanh"),
        input_shape=(10, 1)
    ),
    Dense(16, activation="relu"),
    Dense(1)
])

# 6. Compile model
model.compile(
    optimizer="adam",
    loss="mse"
)

# 7. Train model
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    verbose=1
)

# 8. Make predictions
predictions = model.predict(X_test, verbose=0)

# 9. Convert predictions to original scale
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test)

# 10. Evaluate model
mse = mean_squared_error(actual, predictions)

print("\nBidirectional RNN Results")
print("Mean Squared Error:", mse)

# 11. Visualize results
plt.figure(figsize=(10, 5))

plt.plot(actual, label="Actual Consumption")
plt.plot(predictions, label="Predicted Consumption")

plt.title("Electricity Consumption Forecasting using BiRNN")
plt.xlabel("Time")
plt.ylabel("Electricity Consumption")

plt.legend()
plt.grid()
plt.show()
