import pandas as pd

# ==========================================
# DAY 4 - PANDAS DATA ANALYSIS
# ==========================================

# 1. Load the dataset
df = pd.read_csv("students.csv")

# 2. Display first 5 rows
print("First 5 rows:")
print(df.head())

# 3. Display last 3 rows
print("\nLast 3 rows:")
print(df.tail(3))

# 4. Find number of rows and columns
print("\nShape:")
print(df.shape)

# 5. Display column names
print("\nColumns:")
print(df.columns)

# 6. Display data types and information
print("\nData information:")
df.info()

# 7. Statistical summary
print("\nStatistical summary:")
print(df.describe())


# ==========================================
# FILTERING
# ==========================================

# 8. Students whose marks are greater than 80
print("\nStudents with marks > 80:")
print(df[df["Marks"] > 80])

# 9. Students from Pune
print("\nStudents from Pune:")
print(df[df["City"] == "Pune"])

# 10. AIML students
print("\nAIML students:")
print(df[df["Department"] == "AIML"])

# 11. Students whose age is >= 22
print("\nStudents age >= 22:")
print(df[df["Age"] >= 22])


# ==========================================
# SORTING
# ==========================================

# 12. Sort students by marks
print("\nStudents sorted by marks:")
print(df.sort_values("Marks"))

# 13. Top 3 students by marks
print("\nTop 3 students:")
print(df.sort_values("Marks", ascending=False).head(3))


# ==========================================
# DATA ANALYSIS
# ==========================================

# 14. Average marks
print("\nAverage marks:")
print(df["Marks"].mean())

# 15. Highest marks
print("\nHighest marks:")
print(df["Marks"].max())

# 16. Lowest marks
print("\nLowest marks:")
print(df["Marks"].min())

# 17. Average marks by department
print("\nAverage marks by department:")
print(df.groupby("Department")["Marks"].mean())

# 18. Number of students in each city
print("\nStudents in each city:")
print(df["City"].value_counts())

# 19. Number of students in each department
print("\nStudents in each department:")
print(df["Department"].value_counts())


# ==========================================
# MISSING DATA
# ==========================================

# 20. Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Fill missing Marks with the average marks
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())

print("\nDataset after filling missing values:")
print(df)

# Check again
print("\nMissing values after cleaning:")
print(df.isnull().sum())