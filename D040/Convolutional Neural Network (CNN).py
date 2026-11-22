
# Day 40: Convolutional Neural Network (CNN)
# Image Classification using TensorFlow and Keras

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models

# 1. Load Fashion MNIST Dataset
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress",
    "Coat", "Sandal", "Shirt", "Sneaker",
    "Bag", "Ankle boot"
]

print("Training Data Shape:", x_train.shape)
print("Testing Data Shape:", x_test.shape)

# 2. Normalize Images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# 3. Build CNN Model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    # Convolution Layer
    layers.Conv2D(
        32, (3, 3),
        activation="relu"
    ),

    # Pooling Layer
    layers.MaxPooling2D((2, 2)),

    # Second Convolution Layer
    layers.Conv2D(
        64, (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # Convert Feature Maps into Vector
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(128, activation="relu"),

    # Dropout
    layers.Dropout(0.3),

    # Output Layer
    layers.Dense(10, activation="softmax")
])

# 4. Compile Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Display Model Architecture
model.summary()

# 6. Train CNN Model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# 7. Evaluate Model
test_loss, test_accuracy = model.evaluate(
    x_test, y_test, verbose=0
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# 8. Make Predictions
predictions = model.predict(x_test[:10])

predicted_labels = np.argmax(predictions, axis=1)

print("\nImage Predictions:")

for i in range(10):
    print(
        f"Actual: {class_names[y_test[i]]} | "
        f"Predicted: {class_names[predicted_labels[i]]}"
    )

# 9. Visualize Predictions
plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.imshow(x_test[i].squeeze(), cmap="gray")

    plt.title(
        f"Actual: {class_names[y_test[i]]}\n"
        f"Predicted: {class_names[predicted_labels[i]]}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()

# 10. Plot Training and Validation Accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 11. Save Trained Model
model.save("cnn_image_classifier.keras")

print("\nModel saved successfully!")
