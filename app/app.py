from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Initialize app
app = Flask(__name__)

# Load trained model
model = joblib.load("models/model.pkl")

# Home route
@app.route("/", methods=["GET"])
def home():
    return "Salary Predictor API is running 🚀"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        experience = data["experience"]

        # Convert to numpy
        input_data = np.array([[experience]])

        # Predict
        prediction = model.predict(input_data)[0]

        return jsonify({
            "experience": experience,
            "predicted_salary": float(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)})




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)