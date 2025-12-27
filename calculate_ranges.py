
import pandas as pd
import joblib
import os

def calculate_ranges():
    base_dir = os.getcwd()
    data_path = os.path.join(base_dir, 'data', 'processed', 'X_train.pkl')
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    try:
        X_train = joblib.load(data_path)
        
        # Ensure it's a DataFrame
        if not isinstance(X_train, pd.DataFrame):
            # If it's a numpy array, we might need column names from somewhere or assume order
            # Based on app.py, FEATURE_NAMES = ['Time', 'Amount', 'V4', 'V10', 'V12', 'V14', 'V17', 'V18']
            # But X_train might have all features (V1-V28). 
            # Let's check the shape and type first.
            print(f"Type: {type(X_train)}")
            if hasattr(X_train, 'shape'):
                print(f"Shape: {X_train.shape}")
            
            # If it's a DataFrame, printing describe() is easiest
            if isinstance(X_train, pd.DataFrame):
                 # Filter only relevant columns if they exist
                feature_names = ['Time', 'Amount', 'V4', 'V10', 'V12', 'V14', 'V17', 'V18']
                
                # Check if columns present
                available_cols = [c for c in feature_names if c in X_train.columns]
                
                print(X_train[available_cols].agg(['min', 'max']))
            else:
                print("X_train is not a DataFrame. Please check the data format.")
                # If it's a numpy array, we can't easily guess columns without more info.
                # But typically trained data might be processed to only include selected features?
                # or maybe it's the full dataset.
                
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    calculate_ranges()
