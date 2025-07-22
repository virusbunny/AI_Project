from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Load model, label encoders, and scaler
model = pickle.load(open("model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running. Use POST /predict with JSON input."})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive input JSON
        data = request.get_json()

        # Map sleep duration to numeric
        sleep_map = {
            "Less than 5": 4,
            "5-6 hours": 5.5,
            "6-7 hours": 6.5,
            "7-8 hours": 7.5,
            "More than 8": 9
        }
        data['Sleep Duration'] = sleep_map.get(data['Sleep Duration'], 5.5)

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Label encode categorical columns
        for col, encoder in label_encoders.items():
            if col == 'Profession':
                continue
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col])

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]  # probability of class 1 (Depression)
        return jsonify({'prediction': int(prediction), 'probability': float(probability)})

        

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
