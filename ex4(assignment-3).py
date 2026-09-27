import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# --- 1. Non-Linear Dataset ---
data = {
    "Area_sqft": [800, 1000, 1200, 1500, 1800, 2100, 2500, 3000, 3500, 4000],
    "Price": [
        150000,
        180000,
        220000,
        300000,
        410000,
        530000,
        700000,
        950000,
        1250000,
        1600000,
    ],
}
df = pd.DataFrame(data)

X = df[["Area_sqft"]]
y = df["Price"]

# --- 2. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 3. Linear Regression ---
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
r2_lin = r2_score(y_test, y_pred_lin)

# --- 4. Polynomial Regression (Degree = 2) ---
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# --- 5. Display R2 Comparison ---
print("--- R2 Score Comparison ---")
print(f"Linear Regression R2 Score     : {r2_lin:.4f}")
print(f"Polynomial Regression R2 Score : {r2_poly:.4f}")

# --- 6. Plotting ---
X_seq = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
X_seq_poly = poly.transform(X_seq)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="black", label="Actual Data")
plt.plot(
    X_seq,
    lin_reg.predict(X_seq),
    color="blue",
    linestyle="--",
    label=f"Linear Fit (R2 = {r2_lin:.2f})",
)
plt.plot(
    X_seq,
    poly_reg.predict(X_seq_poly),
    color="red",
    linewidth=2,
    label=f"Polynomial Fit (R2 = {r2_poly:.2f})",
)
plt.xlabel("House Area (sq ft)")
plt.ylabel("Price ($)")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()