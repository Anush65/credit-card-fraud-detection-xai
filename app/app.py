from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load Model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fraud_model.pkl')

model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print("Model not found. Please train the model first.")

load_model()

# Feature names tailored to the training script
FEATURE_NAMES = ['Time', 'Amount', 'V4', 'V10', 'V12', 'V14', 'V17', 'V18']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded. Train the model first.'})

    try:
        # Get data from form
        data = request.form.to_dict()
        
        # Convert to float and list
        input_data = [float(data[f]) for f in FEATURE_NAMES]
        
        # Convert to DataFrame for feature names matching (optional but good for RF)
        input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        
        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] # Prob of class 1 (Fraud)
        
        result = "Fraud" if prediction == 1 else "Legitimate"
        confidence = round(probability * 100, 2)
        
        # Explanation (Feature Importance from the global model)
        # Note: Local interpretability (LIME/SHAP) is better for *why this specific transaction*, 
        # but for simplicity/speed/requirements we show global top features or maybe contribution?
        # User asked "Explain why *a* transaction was flagged".
        # RF doesn't give simple per-instance coefficients like LR.
        # Efficient way: Show the values of the top features compared to average fraud values?
        # Or just show the global feature importance. 
        # Requirement: "Show feature importance... Explain why a transaction was flagged".
        # Let's show Global Importance and the input values for the top features.
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_features = []
        for i in range(min(5, len(importances))):
            idx = indices[i]
            feat_name = FEATURE_NAMES[idx]
            top_features.append({
                'feature': feat_name,
                'importance': round(importances[idx], 4),
                'value': input_data[idx]
            })
            
        return jsonify({
            'result': result,
            'confidence': confidence,
            'top_features': top_features
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
