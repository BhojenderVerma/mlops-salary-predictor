import joblib
import numpy as np

# Load model
model = joblib.load("models/model.pkl")

# Example prediction
experience = np.array([[3]])

prediction = model.predict(experience)

print(f"Predicted salary: {prediction[0]}")