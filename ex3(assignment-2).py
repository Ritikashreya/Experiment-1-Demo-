import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# 1. Mock dataset with missing values
data = {
    "Age": [20, 21, np.nan, 23, 24],
    "Income": [25000, 30000, 28000, np.nan, 40000],
    "City": ["Kolkata", "Durgapur", "Kolkata", "Asansol", "Durgapur"],
    "Purchased": [0, 1, 1, 0, 1],
}

df = pd.DataFrame(data)

X = df.drop("Purchased", axis=1)
y = df["Purchased"]

numeric_features = ["Age", "Income"]
categorical_features = ["City"]

# 2. Preprocessing Pipeline using MinMaxScaler instead of StandardScaler
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),  # Changed to MinMaxScaler
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# 3. Combine with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 4. Apply transformations and split
X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print("--- Processed Feature Matrix (Scaled between 0 and 1) ---")
print(X_processed)