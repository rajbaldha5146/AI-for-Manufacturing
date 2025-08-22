#!/usr/bin/env python3
"""
Script to retrain the model with consistent feature naming.
Run this to fix the feature name mismatch error.
"""

import pandas as pd
from utils.data_generator import generate_training_data
from utils.model_trainer import train_and_save_model

def retrain_model():
    print("🚀 Retraining model with consistent feature names...")
    
    # Generate fresh training data
    print("📊 Generating training data...")
    df = generate_training_data("data/dairy_products.csv", n_samples=1000)
    df.to_csv("data/training_data.csv", index=False)
    print(f"✅ Training data saved with {len(df)} samples")
    
    # Train and save model with feature names
    print("🤖 Training model...")
    train_and_save_model(df, "model/temp_model.pkl")
    print("✅ Model retrained and saved with feature names!")
    
    # Verify the model format
    import joblib
    model_data = joblib.load("model/temp_model.pkl")
    if isinstance(model_data, dict) and 'feature_names' in model_data:
        print(f"🎯 Model saved with {len(model_data['feature_names'])} features:")
        for feature in model_data['feature_names']:
            print(f"   - {feature}")
    else:
        print("⚠️ Warning: Model may not have feature names saved")

if __name__ == "__main__":
    retrain_model()