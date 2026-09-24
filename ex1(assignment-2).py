import pandas as pd
data={
    "Student_Name" :["Amit", "Riya" ,"Sourav" ,"Aditya","Rahul"],
    "Roll_No" : [1,31,49,9,45],
    "Marks"   : [65,72,91,74,65],
    "Attendance" : [81,47,81,19,55]
}
df=pd.DataFrame(data)
print("Students who secured marks above 80: ")
print(df[df["Marks"]>80])