import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# 1. Synthetic dataset with missing values
data = {
    "Age": [20, 21, np.nan, 23, 24],
    "Salary": [25000, 30000, 28000, np.nan, 40000],
    "Department": ["HR", "IT", "HR", "Finance", "IT"],
    "Years_of_Experience": [1, 3, 2, 4, np.nan],
}

df = pd.DataFrame(data)

num_features = ["Age", "Salary", "Years_of_Experience"]
cat_features = ["Department"]

# Categorical transformer
cat_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# 2. Pipeline using StandardScaler
std_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            num_features,
        ),
        ("cat", cat_transformer, cat_features),
    ]
)

# 3. Pipeline using MinMaxScaler
minmax_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", MinMaxScaler()),
                ]
            ),
            num_features,
        ),
        ("cat", cat_transformer, cat_features),
    ]
)

# Apply Transformations
array_std = std_preprocessor.fit_transform(df)
array_minmax = minmax_preprocessor.fit_transform(df)

# Print transformed numeric features
print("--- Transformed Array (StandardScaler) ---")
print(np.round(array_std[:, :3], 4))
print(f"Min Value: {array_std[:, :3].min():.4f}")
print(f"Max Value: {array_std[:, :3].max():.4f}")

print("\n" + "=" * 50 + "\n")

print("--- Transformed Array (MinMaxScaler) ---")
print(np.round(array_minmax[:, :3], 4))
print(f"Min Value: {array_minmax[:, :3].min():.4f}")
print(f"Max Value: {array_minmax[:, :3].max():.4f}")