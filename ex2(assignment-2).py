import pandas as pd

data = {
    "Student Name": ["Aman", "Riya", "Karan", "Neha", "Rahul"],
    "Marks": [72, 92, 67, 77, 74],
    "Attendance": [66, 92, 76, 90, 80]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

print("\nAverage Marks:", df["Marks"].mean())
print("Highest Marks:", df["Marks"].max())
print("Lowest Marks:", df["Marks"].min())
print("Average Attendance:", df["Attendance"].mean())