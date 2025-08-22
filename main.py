from utils.data_generator import generate_training_data
from utils.model_trainer import train_and_save_model

DATASET_PATH = "data/dairy_products.csv"
TRAINING_DATA_PATH = "data/training_data.csv"
MODEL_PATH = "model/temp_model.pkl"

if __name__ == "__main__":
    print("🚀 Generating training data...")
    df = generate_training_data(DATASET_PATH, n_samples=1000)
    df.to_csv(TRAINING_DATA_PATH, index=False)
    print(f"✅ Training data saved to {TRAINING_DATA_PATH}")
    
    print("🤖 Training model...")
    train_and_save_model(df, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
