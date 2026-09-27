import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

# 1. Generate Custom Student Dataset & Train Model
np.random.seed(42)
n_samples = 300
study_hours = np.random.uniform(1, 10, n_samples)
attendance = np.random.uniform(40, 100, n_samples)

underlying_score = (study_hours * 0.6) + (attendance * 0.05) + np.random.normal(0, 0.4, n_samples)
pass_fail = (underlying_score > 6.0).astype(int)

X = np.column_stack((study_hours, attendance))
y = pass_fail

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 2. Get Predicted Probabilities for ROC Curve
y_probs = model.predict_proba(X_test_scaled)[:, 1]

# 3. Calculate ROC Curve and AUC Score
fpr, tpr, _ = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

print(f"Area Under the Curve (AUC) Score: {auc_score:.4f}")

# 4. Plot the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess Line')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()