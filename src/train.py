import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import joblib
import os
import mlflow
import mlflow.sklearn

# 🔥 CHANGE MODEL TYPE HERE
MODEL_TYPE = "linear"   # "linear" or "ridge"

# Load data
data = pd.read_csv("data/salary_data.csv")
X = data[["experience"]]
y = data["salary"]

mlflow.set_experiment("salary-predictor")

with mlflow.start_run():

    # Select model
    if MODEL_TYPE == "linear":
        model = LinearRegression()
        mlflow.log_param("model_type", "LinearRegression")

    elif MODEL_TYPE == "ridge":
        model = Ridge(alpha=1.0)
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("alpha", 1.0)

    # Train
    model.fit(X, y)

    # Predict
    predictions = model.predict(X)

    # Metric
    mse = mean_squared_error(y, predictions)
    mlflow.log_metric("mse", mse)

    print(f"Model: {MODEL_TYPE}")
    print(f"MSE: {mse}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    # Log model
    mlflow.sklearn.log_model(model, "model")

print("Done 🚀")