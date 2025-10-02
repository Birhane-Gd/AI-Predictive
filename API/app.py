from flask import Flask, request, jsonify
import joblib
import numpy as np
app = Flask(__name__)
# Load the trained model
model = joblib.load('../notebook/mrandom_forest_model.pkl')
@app.route('/')
def index():
    return "AI Prediction API is running."
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True) 
        features = data['features'] 
        prediction = model.predict([features])
        pred_value = prediction[0].item()
        return jsonify({'Failure prediction': pred_value})
    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(debug=True)
