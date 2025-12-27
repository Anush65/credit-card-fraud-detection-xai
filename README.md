# Credit Card Fraud Detection System

## Project Objective
This project aims to build a complete end-to-end **Credit Card Fraud Detection System** using **Machine Learning** and **Explainable AI (XAI)**. It features a modern web dashboard for real-time fraud prediction, designed to be user-friendly and suitable for educational demonstrations.

The system:
- Detects fraudulent transactions using a **Random Forest Classifier**.
- Handles class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique).
- Explains predictions using **Feature Importance** (XAI).
- Provides a clean, modern **Flask Web Interface**.

## Core Features
- **Machine Learning**: 
    - Data Cleaning & Preprocessing.
    - Imbalance Handling with SMOTE.
    - Models: Logistic Regression, Decision Tree, Random Forest.
    - Best Model Selected: Random Forest.
- **Explainable AI**:
    - Highlights top contributing features for each prediction.
- **Web Dashboard**:
    - Input form for critical transaction features.
    - One-click "Auto-fill" for demo scenarios.
    - Real-time Fraud/Legitimate classification.
    - Confidence score display.

## Tech Stack
- **Backend API**: Python, Flask
- **Machine Learning**: Pandas, NumPy, Scikit-learn, Imbalanced-learn, Joblib
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3 (Modern UI), JavaScript (Minimal)

## Project Structure
```
credit-card-fraud-detection/
│
├── app/
│   ├── app.py              # Flask Application Entry Point
│   ├── templates/
│   │   └── index.html      # Frontend Dashboard
│   └── static/
│       └── style.css       # Styling
│
├── notebooks/              # Jupyter Notebooks for analysis
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_xai_analysis.ipynb
│
├── models/
│   └── fraud_model.pkl     # Trained Random Forest Model
│
├── data/
│   └── raw/                # Dataset (creditcard.csv)
│
├── src/
│   └── train_model.py      # Script to train and save the model
│
├── requirements.txt        # Python Dependencies
└── README.md               # Project Documentation
```

## How to Run
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**:
   (Optional if `models/fraud_model.pkl` already exists)
   ```bash
   python src/train_model.py
   ```

3. **Run the Web App**:
   ```bash
   python app/app.py
   ```

4. **Access the Dashboard**:
   Open your browser and navigate to `http://127.0.0.1:5000/`.

## Features Used for Prediction
To keep the inputs user-friendly, the system uses the top 8 most important features:
- **Time**: Time elapsed since the first transaction.
- **Amount**: Transaction amount.
- **V4, V10, V12, V14, V17, V18**: Principal components from the original dataset (PCA features).

## Future Scope
- **Deep Learning**: Implement LSTM or Autoencoders for anomaly detection.
- **Real-time API**: Deploy on cloud (AWS/Heroku) for live transaction monitoring.
- **Advanced XAI**: Integrate SHAP (SHapley Additive exPlanations) for deeper local interpretability.
- **Database**: Store prediction history in a SQL/NoSQL database.

---
**Data Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
