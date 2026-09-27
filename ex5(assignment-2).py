import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

# 1. Generate Custom Student Dataset
np.random.seed(42)
n_samples = 300
study_hours = np.random.uniform(1, 10, n_samples)
attendance = np.random.uniform(40, 100, n_samples)

underlying_score = (study_hours * 0.6) + (attendance * 0.05) + np.random.normal(0, 0.4, n_samples)
pass_fail = (underlying_score > 6.0).astype(int)

X = np.column_stack((study_hours, attendance))
y = pass_fail

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model Training
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Threshold Comparison (Your current code)
y_probs = model.predict_proba(X_test_scaled)[:, 1]

thresholds = [0.3, 0.5, 0.7]

print("--- Threshold Comparison ---")
for thresh in thresholds:
    y_pred_custom = (y_probs >= thresh).astype(int)
    
    precision = precision_score(y_test, y_pred_custom, zero_division=0)
    recall = recall_score(y_test, y_pred_custom, zero_division=0)
    
    print(f"Threshold: {thresh} | Precision: {precision:.4f} | Recall: {recall:.4f}")