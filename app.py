from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

base_path = os.path.dirname(os.path.abspath(__file__))

# Load the model and scaler
try:
    model = pickle.load(open(os.path.join(base_path, 'model.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(base_path, 'scaler.pkl'), 'rb'))
except Exception as e:
    print(f"⚠️ Warning: Could not load model files. Run model.py first! Error: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction_data = np.array(data).reshape(1, -1)
    prediction_data = scaler.transform(prediction_data)
    prob = model.predict_proba(prediction_data)[0][1]
    
    advice = "Standard support recommended."
    if prob > 0.6:
        advice = "High Risk: Schedule urgent counseling."
    elif prob > 0.3:
        advice = "Moderate Risk: Monitor progress closely."

    return jsonify({
        'probability': round(prob * 100, 2),
        'advice': advice
    })

if __name__ == '__main__':
    app.run(debug=True)