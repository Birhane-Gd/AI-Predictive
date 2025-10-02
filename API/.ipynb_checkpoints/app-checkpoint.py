from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("AI_Predictive_model.pkl")

@app.route('/')
def index():
    return "AI Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)  # Get JSON data from POST request
        features = data['features']  # Expect input features as a list
        prediction = model.predict([features])  # Predict using your model
        # Convert numpy type to native Python type# int64 is not working in my postman
        pred_value = prediction[0].item()
        return jsonify({'prediction': pred_value})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
