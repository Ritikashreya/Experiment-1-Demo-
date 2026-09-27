import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# 1. Create a synthetic dataset with deliberate missing values
data = {
    "Age": [22, 30, np.nan, 28, 35],
    "Salary": [45000, np.nan, 54000, 60000, 80000],
    "Department": ["HR", "IT", "Finance", np.nan, "IT"],
    "Years_of_Experience": [1.0, 5.0, 3.0, 4.0, np.nan],
}

df = pd.DataFrame(data)

print("--- Original Dataset ---")
print(df)
print("\n" + "=" * 50 + "\n")

# Define numeric and categorical columns
num_features = ["Age", "Salary", "Years_of_Experience"]
cat_features = ["Department"]

# Shared categorical pipeline (Imputer + One-Hot Encoding)
cat_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ]
)

# 2. Pipeline with MinMaxScaler
num_pipeline_minmax = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ]
)

ct_minmax = ColumnTransformer(
    transformers=[
        ("num", num_pipeline_minmax, num_features),
        ("cat", cat_pipeline, cat_features),
    ]
)

# Pipeline with StandardScaler (for comparison)
num_pipeline_std = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

ct_std = ColumnTransformer(
    transformers=[
        ("num", num_pipeline_std, num_features),
        ("cat", cat_pipeline, cat_features),
    ]
)

# Process datasets
transformed_minmax = ct_minmax.fit_transform(df)
transformed_std = ct_std.fit_transform(df)

# 3. Print and compare numerical ranges
print("--- Transformed Array using MinMaxScaler ---")
print(np.round(transformed_minmax, 4))
print(f"\nMin Value: {transformed_minmax.min():.2f}")
print(f"Max Value: {transformed_minmax.max():.2f}")

print("\n" + "=" * 50 + "\n")

print("--- Transformed Array using StandardScaler ---")
print(np.round(transformed_std, 4))
print(f"\nMin Value: {transformed_std.min():.2f}")
print(f"Max Value: {transformed_std.max():.2f}")