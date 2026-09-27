import pandas as pd

data = {
    "Student Name": ["Aman", "Riya", "Karan", "Neha", "Rahul", "Priya"],
    "Roll Number": [1, 32, 23, 14, 37, 27],
    "Marks": [95, 86, 73, 64, 82, 57]
}

df = pd.DataFrame(data)

def calculate_grade(marks):
    if marks >= 90:
        return "A"
    elif marks >= 80:
        return "B"
    elif marks >= 70:
        return "C"
    elif marks >= 60:
        return "D"
    else:
        return "F"

df["Grade"] = df["Marks"].apply(calculate_grade)

print("Student Details with Grades:")
print(df)