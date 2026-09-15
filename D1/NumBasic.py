import numpy as np

# Create marks of 5 students
marks = np.array([78, 85, 92, 67, 88])

print("Marks:", marks)

print("Average:", np.mean(marks))
print("Maximum:", np.max(marks))
print("Minimum:", np.min(marks))
print("Standard Deviation:", np.std(marks))

# Students scoring above average
average = np.mean(marks)

print("\nStudents above average:")
print(marks[marks > average])