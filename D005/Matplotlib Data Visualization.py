import pandas as pd

# ==========================================
# DAY 5 - PANDAS DATA ANALYSIS
# ==========================================

# 1. Create dataset
data = {
    "Name": ["Amit", "Sneha", "Rahul", "Priya", "Vijay"],
    "Age": [22, 24, 21, 23, 25],
    "Marks": [78, 92, 65, 88, 95],
    "City": ["Pune", "Mumbai", "Pune", "Nashik", "Mumbai"]
}

# Create DataFrame
df = pd.DataFrame(data)

print("========== ORIGINAL DATA ==========")
print(df)


# 2. Explore the dataset
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== LAST 5 ROWS ==========")
print(df.tail())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)

print("\n========== DATA INFORMATION ==========")
print(df.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())


# 3. Select columns
print("\n========== NAMES ==========")
print(df["Name"])

print("\n========== MARKS ==========")
print(df["Marks"])

print("\n========== NAME AND MARKS ==========")
print(df[["Name", "Marks"]])


# 4. Filter data
print("\n========== STUDENTS WITH MARKS > 80 ==========")
print(df[df["Marks"] > 80])

print("\n========== STUDENTS WITH MARKS > 70 ==========")
print(df[df["Marks"] > 70])

print("\n========== STUDENTS WITH AGE > 22 ==========")
print(df[df["Age"] > 22])

print("\n========== STUDENTS FROM MUMBAI ==========")
print(df[df["City"] == "Mumbai"])


# 5. Basic analysis
print("\n========== MARKS ANALYSIS ==========")

print("Average Marks:", df["Marks"].mean())
print("Highest Marks:", df["Marks"].max())
print("Lowest Marks:", df["Marks"].min())
print("Total Marks:", df["Marks"].sum())


# 6. Student with highest marks
print("\n========== TOP STUDENT ==========")

top_student = df.loc[df["Marks"].idxmax()]
print(top_student)


# 7. Count students with marks above 80
print("\n========== STUDENTS ABOVE 80 ==========")

students_above_80 = df[df["Marks"] > 80]
print("Number of students:", len(students_above_80))


# 8. Add Result column
df["Result"] = df["Marks"].apply(
    lambda marks: "Pass" if marks >= 40 else "Fail"
)

print("\n========== DATA WITH RESULT ==========")
print(df)


# 9. Count passed students
print("\n========== PASS COUNT ==========")

passed_students = df[df["Result"] == "Pass"]
print("Number of students passed:", len(passed_students))


# 10. Final summary
print("\n========================================")
print("           DAY 5 SUMMARY")
print("========================================")

print("Total Students:", len(df))
print("Average Marks:", df["Marks"].mean())
print("Highest Marks:", df["Marks"].max())
print("Students Above 80:", len(students_above_80))
print("Mumbai Students:", len(df[df["City"] == "Mumbai"]))
print("Students Passed:", len(passed_students))

print("\nDay 5 Pandas Data Analysis Completed!")