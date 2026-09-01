#!/usr/bin/env python3
import os
import joblib
import pandas as pd
import numpy as np

def calculate_mcnemar():
    print("Loading test dataset...")
    test_file = "dataset/test_features.csv"
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found.")
        return
    
    df = pd.read_csv(test_file)
    y_true = df["label"].values
    
    with open("models/feature_names.json", "r") as f:
        import json
        feature_names = json.load(f)
        
    X_test = df[feature_names]
    
    scaler = joblib.load("models/scaler.joblib")
    X_scaled = scaler.transform(X_test)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    
    print("Loading models...")
    rf_model = joblib.load("models/random_forest_model.joblib")
    svm_model = joblib.load("models/svm_model.joblib")
    
    print("Predicting with RF...")
    rf_preds = rf_model.predict(X_scaled)
    
    print("Predicting with SVM...")
    svm_preds = svm_model.predict(X_scaled)
    
    rf_correct = (rf_preds == y_true)
    svm_correct = (svm_preds == y_true)
    
    b = np.sum(rf_correct & ~svm_correct)
    c = np.sum(~rf_correct & svm_correct)
    
    print(f"n = {len(y_true)}")
    print(f"RF correct, SVM wrong (b): {b}")
    print(f"RF wrong, SVM correct (c): {c}")
    
    # McNemar's with continuity correction
    chi2 = ((abs(b - c) - 1.0) ** 2) / (b + c)
    print(f"Calculated McNemar chi2: {chi2:.2f}")

if __name__ == "__main__":
    calculate_mcnemar()
