import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --- 1. Dataset Creation ---
data = {
    "Area_sqft": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500],
    "Bedrooms": [2, 2, 3, 3, 4, 3, 4, 4, 5, 5],
    "Age_years": [10, 8, 5, 12, 3, 15, 2, 6, 1, 4],
    "Price": [
        200000,
        230000,
        280000,
        320000,
        360000,
        390000,
        450000,
        490000,
        530000,
        610000,
    ],
}
df = pd.DataFrame(data)

X = df[["Area_sqft", "Bedrooms", "Age_years"]]
y = df["Price"]

# --- 2. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 3. Model Training ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- 4. Prediction & Evaluation ---
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"  {col}: {coef:.2f}")
print("Intercept:", round(model.intercept_, 2))

print("\n--- Evaluation Metrics ---")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")