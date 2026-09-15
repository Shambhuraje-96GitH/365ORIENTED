import pandas as pd

# Create a DataFrame
data = {
    "Name": ["Rahul", "Priya", "Amit", "Sneha"],
    "Age": [21, 22, 20, 23],
    "Score": [85, 92, 78, 95]
}

df = pd.DataFrame(data)

# Display the data
print("Student Data:")
print(df)

# Display first rows
print("\nFirst 2 Rows:")
print(df.head(2))

# Display information
print("\nData Information:")
print(df.info())

# Basic statistics
print("\nStatistics:")
print(df.describe())

# Select a column
print("\nStudent Names:")
print(df["Name"])

# Filter students with score greater than 85
print("\nStudents with Score > 85:")
print(df[df["Score"] > 85])

# Calculate average score
print("\nAverage Score:")
print(df["Score"].mean())