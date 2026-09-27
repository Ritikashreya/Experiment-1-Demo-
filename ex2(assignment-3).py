import pandas as pd

data = {
    "Student Name": ["Aman", "Riya", "Karan", "Neha", "Rahul"],
    "Marks": [72, 92, 67, 77, 74],
    "Attendance": [66, 92, 76, 90, 80]
}

df = pd.DataFrame(data)

df["Result"] = df["Marks"].apply(lambda x: "Pass" if x >= 40 else "Fail")

print("Student Data:")
print(df)

print("\nPassed Students:")
print(df[df["Result"] == "Pass"])

print("\nFailed Students:")
print(df[df["Result"] == "Fail"])