#!/usr/bin/env python3
"""
Test script to verify the feature name mismatch fix.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from utils.model_utils import load_model, predict_temperature, prepare_features
from utils.model_trainer import evaluate_model

def test_feature_consistency():
    print("🧪 Testing feature name consistency fix...")
    
    try:
        # Load model and feature names
        print("📂 Loading model...")
        model, feature_names = load_model("model/temp_model.pkl")
        print(f"✅ Model loaded with {len(feature_names)} feature names")
        
        # Load training data
        print("📊 Loading training data...")
        training_df = pd.read_csv("data/training_data.csv")
        X = training_df.drop(columns=['recommended_temp'])
        y = training_df['recommended_temp']
        
        # Split data
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Prepare features
        print("🔧 Preparing features...")
        X_test_prepared = prepare_features(X_test, feature_names)
        print(f"✅ Features prepared: {list(X_test_prepared.columns)}")
        
        # Test evaluation
        print("📈 Testing model evaluation...")
        metrics = evaluate_model(model, X_test_prepared, y_test, feature_names)
        print(f"✅ Model evaluation successful!")
        print(f"   MAE: {metrics['mae']:.3f}°C")
        print(f"   RMSE: {metrics['rmse']:.3f}°C")
        print(f"   R²: {metrics['r2_score']:.3f}")
        
        # Test prediction
        print("🎯 Testing prediction...")
        products_df = pd.read_csv("data/dairy_products.csv")
        product_list = products_df["product"].tolist()
        
        test_products = ["Milk", "Cheese"]
        test_temp = 25.0
        
        result = predict_temperature(model, test_products, test_temp, product_list, feature_names)
        print(f"✅ Prediction successful: {result:.2f}°C for {test_products} at {test_temp}°C external")
        
        print("\n🎉 All tests passed! Feature name mismatch is fixed.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_feature_consistency()