from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from tensorflow import keras

# Load saved artifacts
model = keras.models.load_model("ml-backend/model/model_nn.keras")
scaler = joblib.load("ml-backend/model/scaler.pkl")
feature_means = joblib.load("ml-backend/model/feature_means.pkl")

# Flask app
app = Flask(__name__)
CORS(app)  # Allow requests from React frontend

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    age = float(data['age'])
    bmi = float(data['bmi'])
    children = float(data['children'])
    smoker = 1 if data['smoker'].lower() == 'yes' else 0

    input_df = pd.DataFrame([feature_means])
    input_df.at[0, 'age'] = age
    input_df.at[0, 'bmi'] = bmi
    input_df.at[0, 'children'] = children
    input_df.at[0, 'smoker'] = smoker

    input_scaled = scaler.transform(input_df)
    result = model.predict(input_scaled)[0]

    result = float(model.predict(input_scaled)[0][0])
    return jsonify({'charges': round(result, 2)})

# Run the function
if __name__ == '__main__':
    app.run(debug=True)