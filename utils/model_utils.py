import joblib
import pandas as pd

def load_model(path):
    """Load model and return both model and feature names."""
    model_data = joblib.load(path)
    
    # Handle both old and new model formats
    if isinstance(model_data, dict) and 'model' in model_data:
        return model_data['model'], model_data.get('feature_names', [])
    else:
        # Old format - just the model
        return model_data, []

def prepare_features(df, feature_names):
    """
    Prepare dataframe features to match training feature names.
    
    Args:
        df: Input dataframe
        feature_names: Expected feature names from training
    
    Returns:
        DataFrame with consistent feature names and order
    """
    if not feature_names:
        return df
    
    # Create a new dataframe with expected features
    prepared_df = pd.DataFrame(columns=feature_names)
    
    # Copy existing columns that match
    for col in feature_names:
        if col in df.columns:
            prepared_df[col] = df[col]
        else:
            # Fill missing features with 0
            prepared_df[col] = 0
    
    return prepared_df

def predict_temperature(model, selected_products, external_temp, product_list, feature_names=None):
    """
    Predict temperature with consistent feature naming.
    
    Args:
        model: Trained ML model
        selected_products: List of selected dairy products
        external_temp: External temperature
        product_list: List of all available products
        feature_names: Expected feature names from training
    
    Returns:
        Predicted temperature
    """
    # Create features with "Has_" prefix to match training data
    data = {f"Has_{p}": (1 if p in selected_products else 0) for p in product_list}
    data["external_temp"] = external_temp
    df = pd.DataFrame([data])
    
    # Ensure feature consistency if feature_names provided
    if feature_names:
        df = prepare_features(df, feature_names)
    
    return model.predict(df)[0]
