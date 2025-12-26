import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'creditcard.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fraud_model.pkl')

# Features selected for the web app
SELECTED_FEATURES = ['Time', 'Amount', 'V4', 'V10', 'V12', 'V14', 'V17', 'V18']
TARGET = 'Class'

def train():
    print("Loading data in chunks to save memory...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    # Chunk processing: Keep all Class 1, Downsample Class 0
    chunksize = 50000
    df_list = []
    
    try:
        for chunk in pd.read_csv(DATA_PATH, chunksize=chunksize, usecols=SELECTED_FEATURES + [TARGET]):
            fraud = chunk[chunk[TARGET] == 1]
            legit = chunk[chunk[TARGET] == 0].sample(frac=0.1, random_state=42) # Keep 10% of legit
            df_list.append(pd.concat([fraud, legit]))
            print(f"Processed chunk, frauds found: {len(fraud)}")
        
        df = pd.concat(df_list)
        print(f"Final Dataset shape: {df.shape}")
        
        # Select features
        X = df[SELECTED_FEATURES]
        y = df[TARGET]

        print(f"Features: {SELECTED_FEATURES}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Handle Imbalance using SMOTE
        print("Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"Resampled Class distribution: {y_train_resampled.value_counts().to_dict()}")

        # Train Random Forest
        print("Training Random Forest Model...")
        rf = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42, n_jobs=1)
        rf.fit(X_train_resampled, y_train_resampled)

        # Evaluate
        print("Evaluating model...")
        y_pred = rf.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

        # Save model
        print(f"Saving model to {MODEL_PATH}...")
        joblib.dump(rf, MODEL_PATH)
        print("Done!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    train()
