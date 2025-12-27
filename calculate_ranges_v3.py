
import pandas as pd
import os

def calculate_ranges():
    base_dir = os.getcwd()
    data_path = os.path.join(base_dir, 'data', 'raw', 'creditcard.csv')
    output_file = os.path.join(base_dir, 'features_ranges.txt')
    
    if not os.path.exists(data_path):
        with open(output_file, 'w') as f:
            f.write(f"Error: {data_path} not found.")
        return

    features = ['Time', 'Amount', 'V4', 'V10', 'V12', 'V14', 'V17', 'V18']
    
    try:
        # Load only necessary columns
        df = pd.read_csv(data_path, usecols=features)
        metrics = df.agg(['min', 'max'])
        
        with open(output_file, 'w') as f:
            for col in features:
                f.write(f"{col}: {metrics[col]['min']} to {metrics[col]['max']}\n")
        
    except Exception as e:
        with open(output_file, 'w') as f:
            f.write(f"An error occurred: {e}")

if __name__ == "__main__":
    calculate_ranges()
