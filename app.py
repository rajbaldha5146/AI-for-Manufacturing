import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from utils.model_utils import load_model, predict_temperature, prepare_features
from utils.model_trainer import evaluate_model, get_feature_importance

st.set_page_config(page_title="Smart Dairy Temp Controller", layout="centered")

# Load resources
products_df = pd.read_csv("data/dairy_products.csv")
product_list = products_df["product"].tolist()
model, feature_names = load_model("model/temp_model.pkl")

# Load training data and evaluate model accuracy
@st.cache_data
def get_model_data():
    try:
        training_df = pd.read_csv("data/training_data.csv")
        X = training_df.drop(columns=['recommended_temp'])
        y = training_df['recommended_temp']
        
        # Split data with same random state as training
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Prepare features to match training format
        if feature_names:
            X_test = prepare_features(X_test, feature_names)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, feature_names)
        
        # Get predictions for visualization
        predictions = model.predict(X_test)
        
        return {
            'metrics': metrics,
            'X_test': X_test,
            'y_test': y_test,
            'predictions': predictions
        }
    except Exception as e:
        st.error(f"Could not load model data: {e}")
        return None

@st.cache_data
def get_feature_importance_data():
    try:
        if feature_names:
            importance_df = get_feature_importance(model, feature_names)
            return importance_df
        return None
    except Exception as e:
        st.warning(f"Could not compute feature importance: {e}")
        return None

model_data = get_model_data()
importance_data = get_feature_importance_data()


st.title("🥛 Smart Dairy Room Temperature Controller")
st.markdown("Use this app to get the **ideal room temperature** for storing selected dairy items based on the current external temperature.")

# Model Analysis Section with Tabs
if model_data:
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Model Accuracy", "🧠 Explainable AI", "📊 Prediction Visualization"])
    
    with tab1:
        st.subheader("Model Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Mean Absolute Error",
                value=f"{model_data['metrics']['mae']:.2f}°C",
                help="Average prediction error in degrees Celsius"
            )
        
        with col2:
            st.metric(
                label="Root Mean Squared Error", 
                value=f"{model_data['metrics']['rmse']:.2f}°C",
                help="Standard deviation of prediction errors"
            )
        
        with col3:
            st.metric(
                label="R² Score",
                value=f"{model_data['metrics']['r2_score']:.3f}",
                help="Coefficient of determination (1.0 = perfect predictions)"
            )
        
        # Performance interpretation
        if model_data['metrics']['r2_score'] > 0.9:
            st.success("🎉 Excellent model performance!")
        elif model_data['metrics']['r2_score'] > 0.8:
            st.info("👍 Good model performance")
        else:
            st.warning("⚠️ Model could be improved")
    
    with tab2:
        st.subheader("Feature Importance (Explainable AI)")
        
        if importance_data is not None and not importance_data.empty:
            # Create feature importance bar chart
            fig = px.bar(
                importance_data.head(10),  # Show top 10 features
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Most Important Features",
                labels={'importance': 'Importance Score', 'feature': 'Features'},
                color='importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(
                height=500,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show feature importance table
            st.subheader("Feature Importance Details")
            
            # Clean up feature names for display
            display_df = importance_data.copy()
            display_df['feature'] = display_df['feature'].str.replace('Has_', '').str.replace('_', ' ')
            display_df['importance'] = display_df['importance'].round(4)
            
            st.dataframe(
                display_df,
                column_config={
                    "feature": "Feature",
                    "importance": st.column_config.ProgressColumn(
                        "Importance Score",
                        help="Higher values indicate more important features",
                        min_value=0,
                        max_value=display_df['importance'].max(),
                    ),
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Explanation
            st.info("""
            **Understanding Feature Importance:**
            - Higher scores indicate features that have more influence on temperature predictions
            - External temperature and specific dairy products typically have the highest importance
            - This helps explain which factors the AI considers most when making recommendations
            """)
        else:
            st.warning("⚠️ Feature importance data is not available. This may occur if the model hasn't been retrained with the new format.")
    
    with tab3:
        st.subheader("Model Predictions vs Actual Values")
        
        if 'predictions' in model_data and 'y_test' in model_data:
            # Create actual vs predicted scatter plot
            fig = go.Figure()
            
            # Add actual vs predicted line (perfect predictions)
            min_temp = min(model_data['y_test'].min(), model_data['predictions'].min())
            max_temp = max(model_data['y_test'].max(), model_data['predictions'].max())
            
            fig.add_trace(go.Scatter(
                x=[min_temp, max_temp],
                y=[min_temp, max_temp],
                mode='lines',
                name='Perfect Predictions',
                line=dict(color='red', dash='dash'),
                opacity=0.7
            ))
            
            # Add actual vs predicted scatter
            fig.add_trace(go.Scatter(
                x=model_data['y_test'],
                y=model_data['predictions'],
                mode='markers',
                name='Predictions',
                marker=dict(
                    color='blue',
                    size=8,
                    opacity=0.6
                ),
                text=[f"Actual: {a:.1f}°C<br>Predicted: {p:.1f}°C" 
                      for a, p in zip(model_data['y_test'], model_data['predictions'])],
                hovertemplate='%{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Actual vs Predicted Temperatures",
                xaxis_title="Actual Temperature (°C)",
                yaxis_title="Predicted Temperature (°C)",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Time series style plot
            st.subheader("Prediction Timeline")
            
            sample_indices = list(range(len(model_data['y_test'])))
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=sample_indices,
                y=model_data['y_test'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='green', width=2),
                marker=dict(size=4)
            ))
            
            fig2.add_trace(go.Scatter(
                x=sample_indices,
                y=model_data['predictions'],
                mode='lines+markers',
                name='Predicted',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            fig2.update_layout(
                title="Model Predictions vs Actual Values Over Test Samples",
                xaxis_title="Sample Index",
                yaxis_title="Temperature (°C)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Max Error",
                    f"{abs(model_data['y_test'] - model_data['predictions']).max():.2f}°C"
                )
            
            with col2:
                st.metric(
                    "Min Error", 
                    f"{abs(model_data['y_test'] - model_data['predictions']).min():.2f}°C"
                )
            
            with col3:
                correlation = pd.Series(model_data['y_test']).corr(pd.Series(model_data['predictions']))
                st.metric(
                    "Correlation",
                    f"{correlation:.3f}"
                )
        else:
            st.warning("⚠️ Prediction data is not available for visualization.")

st.markdown("---")

st.sidebar.header("🔍 How to Use")
st.sidebar.markdown("""
1. Select one or more dairy items from the list.
2. Enter the current external temperature in °C.
3. Click the button to get the AI-recommended room temperature.
""")

# Sidebar Product Info Table
st.sidebar.markdown("---")
st.sidebar.subheader("🧊 Product Temp Requirements")

for _, row in products_df.iterrows():
    st.sidebar.markdown(
        f"**{row['product'].capitalize()}**: {row['min_temp']}°C to {row['max_temp']}°C"
    )

# Form input
selected_products = st.multiselect("Select Dairy Products", product_list)
ext_temp = st.number_input("External Temperature (°C)", min_value=-30.0, max_value=60.0, value=25.0)

if st.button("🔎 Get Recommended Room Temperature"):
    if not selected_products:
        st.warning("Please select at least one dairy product.")
    else:
        isolated_items = {"Ice cream", "Ghee"}
        selected_set = set(selected_products)

        if isolated_items & selected_set and len(selected_set) > 1:
            conflict_items = isolated_items & selected_set
            st.error(f"❌ These items are not recommended to be stored with others: {', '.join(conflict_items)}")
            
        else:
            result = predict_temperature(model, selected_products, ext_temp, product_list, feature_names)
            st.success(f"✅ Recommended Storage Temperature: **{result:.2f}°C**")