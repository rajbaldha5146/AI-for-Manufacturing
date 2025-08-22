import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_and_save_model(data_df, model_path):
    X = data_df.drop(columns=['recommended_temp'])
    y = data_df['recommended_temp']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"✅ Model MAE on test set: {mae:.2f}°C")

    # Save model with feature names for consistency
    model_data = {
        'model': model,
        'feature_names': list(X.columns)
    }
    joblib.dump(model_data, model_path)

def evaluate_model(model, X_test, y_test, feature_names=None):
    """
    Evaluate model performance and return accuracy metrics.
    
    Args:
        model: Trained ML model
        X_test: Test features
        y_test: Test targets
        feature_names: Expected feature names from training
    
    Returns:
        dict: Dictionary containing MAE, RMSE, and R² score
    """
    # Ensure feature consistency if feature_names provided
    if feature_names is not None:
        X_test = prepare_features_for_evaluation(X_test, feature_names)
    
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2
    }

def prepare_features_for_evaluation(df, expected_features):
    """
    Prepare test features to match training feature names.
    
    Args:
        df: DataFrame with test features
        expected_features: List of feature names from training
    
    Returns:
        DataFrame with consistent feature names and order
    """
    import pandas as pd
    
    # Create a new dataframe with expected features
    prepared_df = pd.DataFrame(columns=expected_features)
    
    # Copy existing columns that match
    for col in expected_features:
        if col in df.columns:
            prepared_df[col] = df[col]
        else:
            # Fill missing features with 0
            prepared_df[col] = 0
    
    return prepared_df

def get_feature_importance(model, feature_names):
    """
    Get feature importance from RandomForest model.
    
    Args:
        model: Trained RandomForest model
        feature_names: List of feature names
    
    Returns:
        pandas.DataFrame: Sorted dataframe with features and importance scores
    """
    import pandas as pd
    
    if not hasattr(model, 'feature_importances_'):
        return None
    
    # Create dataframe with feature names and importance scores
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    # Sort by importance (descending)
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df
