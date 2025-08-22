import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("data/dairy_products.csv")
products = df["product"].tolist()

# Generate synthetic data
samples = []
np.random.seed(42)
for _ in range(1000):
    selected = np.random.choice(products, size=np.random.randint(1, 5), replace=False)
    ext_temp = np.random.uniform(-10, 45)
    min_req = df[df["product"].isin(selected)]["min_temp"].max()
    max_req = df[df["product"].isin(selected)]["max_temp"].min()
    rec_temp = (min_req + max_req + ext_temp * 0.1) / 2
    row = {p: (1 if p in selected else 0) for p in products}
    row["external_temp"] = ext_temp
    row["recommended_temp"] = rec_temp
    samples.append(row)

df_train = pd.DataFrame(samples)
X = df_train.drop(columns=["recommended_temp"])
y = df_train["recommended_temp"]

# Train and save model
model = RandomForestRegressor()
model.fit(X, y)
joblib.dump(model, "model/temp_model.pkl")
print("Model trained and saved as 'model/temp_model.pkl'")