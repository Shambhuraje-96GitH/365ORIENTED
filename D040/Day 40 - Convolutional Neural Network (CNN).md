
# Day 40 - Convolutional Neural Network (CNN)

## Objective
Implement a Convolutional Neural Network using TensorFlow
and Keras for image classification.

## Dataset
Fashion MNIST Dataset.

The dataset contains 70,000 grayscale images of clothing
items in 10 different categories.

- Training images: 60,000
- Testing images: 10,000
- Image dimensions: 28 x 28 pixels
- Number of classes: 10

## Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

## CNN Architecture

1. Input Layer
2. Convolutional Layer (32 filters)
3. Max Pooling Layer
4. Convolutional Layer (64 filters)
5. Max Pooling Layer
6. Flatten Layer
7. Dense Layer (128 neurons)
8. Dropout Layer
9. Output Layer (10 classes)

## Implementation Steps

1. Load Fashion MNIST dataset.
2. Normalize image pixel values.
3. Build CNN architecture.
4. Compile the model using Adam optimizer.
5. Train the model.
6. Evaluate test accuracy.
7. Predict clothing categories.
8. Visualize predictions.
9. Save the trained model.

## Installation

Install the required libraries:

pip install tensorflow numpy matplotlib

## Execution

python Day40_CNN_Image_Classification.py

## Expected Output

- Dataset dimensions
- CNN model architecture
- Training and validation accuracy
- Test loss and accuracy
- Predicted image categories
- Image prediction visualization
- Saved CNN model

## Learning Outcomes

- Understand CNN architecture.
- Learn image feature extraction.
- Understand convolution and pooling.
- Implement deep learning image classification.
- Evaluate and save a trained CNN model.

## Conclusion

Successfully implemented a Convolutional Neural Network
for classifying clothing images using the Fashion MNIST
dataset.
