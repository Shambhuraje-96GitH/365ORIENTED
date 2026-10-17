import numpy as np

# Create an array
numbers = np.array([10, 20, 30, 40, 50])

print("Array:", numbers)

# Array properties
print("Shape:", numbers.shape)
print("Dimensions:", numbers.ndim)

# Indexing
print("First element:", numbers[0])
print("Last element:", numbers[-1])

# Slicing
print("First three elements:", numbers[:3])

# Mathematical operations
print("Addition:", numbers + 5)
print("Multiplication:", numbers * 2)

# Statistical operations
print("Sum:", np.sum(numbers))
print("Mean:", np.mean(numbers))
print("Maximum:", np.max(numbers))
print("Minimum:", np.min(numbers))

# Reshaping
matrix = np.array([1, 2, 3, 4, 5, 6])
matrix = matrix.reshape(2, 3)

print("Reshaped array:")
print(matrix)