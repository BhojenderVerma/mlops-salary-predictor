import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# 1. Load data
data = pd.read_csv("data/salary_data.csv")

# 2. Split into X (input) and y (output)
X = data[["experience"]]   # feature
y = data["salary"]         # target

# 3. Create model
model = LinearRegression()

# 4. Train model
model.fit(X, y)

# 5. Print model details
print("Model trained successfully!")
print(f"Coefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# 6. Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("Model saved at models/model.pkl")