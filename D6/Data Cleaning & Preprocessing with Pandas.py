import pandas as pd


# ==========================================
# DAY 6: DATA CLEANING & PREPROCESSING
# ==========================================

# 1. Create a messy dataset
data = {
    "Name": ["Rahul", "Priya", "Amit", "Sneha", "Rahul", "Vikas"],
    "Age": [21, 22, None, 24, 21, None],
    "Salary": [30000, 45000, 35000, None, 30000, 50000],
    "Department": ["IT", "HR", "IT", "Finance", "IT", "HR"]
}

df = pd.DataFrame(data)


# 2. Display the original dataset
print("========== ORIGINAL DATASET ==========")
print(df)


# 3. Display dataset information
print("\n========== DATASET INFORMATION ==========")
print(df.info())


# 4. Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# 5. Check duplicate rows
print("\n========== DUPLICATE ROWS ==========")
print("Number of duplicate rows:", df.duplicated().sum())


# ==========================================
# 6. HANDLE MISSING AGE VALUES
# ==========================================

average_age = df["Age"].mean()

print("\nAverage Age:", average_age)

df["Age"] = df["Age"].fillna(average_age)


# ==========================================
# 7. HANDLE MISSING SALARY VALUES
# ==========================================

median_salary = df["Salary"].median()

print("Median Salary:", median_salary)

df["Salary"] = df["Salary"].fillna(median_salary)


# ==========================================
# 8. REMOVE DUPLICATE ROWS
# ==========================================

print("\nDuplicates before removing:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicates after removing:", df.duplicated().sum())


# ==========================================
# 9. CHECK DATA TYPES
# ==========================================

print("\n========== DATA TYPES BEFORE CONVERSION ==========")
print(df.dtypes)


# Convert Age to integer
df["Age"] = df["Age"].astype(int)


print("\n========== DATA TYPES AFTER CONVERSION ==========")
print(df.dtypes)


# ==========================================
# 10. FINAL CLEANED DATASET
# ==========================================

print("\n========== CLEANED DATASET ==========")
print(df)


# ==========================================
# 11. VERIFY MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df.isnull().sum())


# ==========================================
# 12. VERIFY DUPLICATES
# ==========================================

print("\n========== DUPLICATES AFTER CLEANING ==========")
print("Number of duplicate rows:", df.duplicated().sum())


# ==========================================
# 13. DAY 6 CHALLENGE
# ==========================================

# Add Experience column
df["Experience"] = [1, 2, 3, 4, 1]

print("\n========== DATASET WITH EXPERIENCE ==========")
print(df)


# Filter employees with more than 2 years of experience
experienced_employees = df[df["Experience"] > 2]

print("\n========== EMPLOYEES WITH MORE THAN 2 YEARS EXPERIENCE ==========")
print(experienced_employees)


# Calculate average salary of employees
# with more than 2 years of experience
average_experienced_salary = experienced_employees["Salary"].mean()

print("\nAverage Salary of Employees with More Than 2 Years Experience:",
      average_experienced_salary)


# ==========================================
# 14. FINAL SUMMARY
# ==========================================

print("\n========== FINAL SUMMARY ==========")
print("Total Employees:", len(df))
print("Average Age:", df["Age"].mean())
print("Average Salary:", df["Salary"].mean())
print("Average Salary (>2 Years Experience):",
      average_experienced_salary)