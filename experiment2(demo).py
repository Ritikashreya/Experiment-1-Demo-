import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.datasets import load_iris

#1. load dataset
iris=load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

# 2.basic data exploration
print("--- First five rows ---")
print(df.head())

print("\n--- dataset information ---")
print(df.info())

print("\n--- statistical summary ---")
print(df.describe())

print("\n--- missing values ---")
print(df.isnull().sum())

#3.correlation matrix

print("\n--correlation matrix--")
print(df.corr(numeric_only=True))

#4.scatter plot: sepal length vs petal length

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=df,
    x="sepal length (cm)",
    y="petal length (cm)",
    hue ="target",
    palette="viridis"
)

plt.title("sepal length vs petal length by class")
plt.show()

#5. histogram : Distribution of sepal length
plt.figure(figsize=(7,5))
sns.histplot(df["sepal length (cm)"],kde = True, color="blue")
plt.title("Distribution of sepal length")
plt.show()