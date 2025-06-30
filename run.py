import pandas as pd
import numpy as np
from preprocess import preprocess_student_data
from train_model import train_dropout_model
import joblib
import os

# Create more robust sample data
data = {
    'gpa': [3.2, 3.8, 2.5, 3.9, 3.1, 2.1, 3.5, 2.9],
    'attendance_rate': [0.9, 0.95, 0.6, 0.88, 0.75, 0.5, 0.8, 0.65],
    'test_scores': [85, 92, 72, 95, 68, 60, 82, 71],
    'socioeconomic_status': ['medium', 'high', 'low', 'high', 'medium', 'low', 'high', 'medium'],
    'school_branch': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'dropout': [0, 0, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

try:
    print("Step 1: Preprocessing data...")
    X = preprocess_student_data(df)
    y = df['dropout'].values
    
    print("Step 2: Training model...")
    model, metrics = train_dropout_model(X, y)
    print(f"Training successful!\nMetrics: {metrics}")
    
    print("Step 3: Saving model...")
    model_path = os.path.join(os.getcwd(), 'dropout_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Verify the file was created
    if os.path.exists(model_path):
        print("Verification: Model file created successfully!")
    else:
        print("Warning: Model file not found!")
        
except Exception as e:
    print(f"\nERROR: {str(e)}")
    print("\nTroubleshooting steps:")
    print("1. Make sure you have installed:")
    print("   pip install scikit-learn pandas numpy joblib")
    print("2. Check all files are in the same directory")
    print("3. Ensure no Python files are named 'sklearn.py'")