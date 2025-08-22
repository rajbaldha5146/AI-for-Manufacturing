import pandas as pd
import random

def generate_training_data(csv_path, n_samples=500):
    df = pd.read_csv(csv_path)
    products = df['product'].tolist()
    records = []

    for _ in range(n_samples):
        n_products = random.randint(1, 4)
        selected = random.sample(products, n_products)
        external_temp = random.uniform(-10, 40)

        filtered = df[df['product'].isin(selected)]
        overall_min = filtered['min_temp'].max()
        overall_max = filtered['max_temp'].min()

        if overall_min > overall_max:
            continue

        recommended = (overall_min + overall_max) / 2
        if external_temp > 25:
            recommended -= 0.5
        elif external_temp < 5:
            recommended += 0.5

        record = {
            'external_temp': external_temp,
            'recommended_temp': recommended,
        }
        for p in products:
            record[f'Has_{p}'] = 1 if p in selected else 0

        records.append(record)

    return pd.DataFrame(records)
