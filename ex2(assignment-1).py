import pandas as pd

data = {
    "Student Name": ["Aman", "Riya", "Karan", "Neha", "Rahul"],
    "Marks": [72, 92, 67, 77, 74],
    "Attendance": [66, 92, 76, 90, 80]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

print("\nSummary Statistics:")
print(df.describe())