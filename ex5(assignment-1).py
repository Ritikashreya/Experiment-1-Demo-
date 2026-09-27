import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Generate Custom Student Dataset
np.random.seed(42)
n_samples = 300
study_hours = np.random.uniform(1, 10, n_samples)
attendance = np.random.uniform(40, 100, n_samples)

# Create a binary target (1 for Pass, 0 for Fail)
underlying_score = (study_hours * 0.6) + (attendance * 0.05) + np.random.normal(0, 0.4, n_samples)
pass_fail = (underlying_score > 6.0).astype(int)

X = np.column_stack((study_hours, attendance))
y = pass_fail

# 2. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model Training
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Initial Prediction & Evaluation
y_pred = model.predict(X_test_scaled)
print(f"Model Accuracy (Default 0.5 Threshold): {accuracy_score(y_test, y_pred):.4f}")