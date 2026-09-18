import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.airtwinnet.components.st_gnn import STGNN


DATA_PATH = "data/processed/air_quality_temporal.csv"
MODEL_PATH = "artifacts/st_gnn_model_v2.pt"

FEATURES = [
    "temperature",
    "humidity",
    "no2",
    "co",
    "o3",
    "hour",
    "day_of_week",
    "month"
]

TARGETS = [
    "pm2_5_future",
    "pm10_future"
]

SEQUENCE_LENGTH = 12


def create_sequences(data):
    X = []
    y = []

    feature_values = data[FEATURES].values
    target_values = data[TARGETS].values

    for i in range(SEQUENCE_LENGTH, len(data)):
        X.append(
            feature_values[i - SEQUENCE_LENGTH:i]
        )
        y.append(target_values[i])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, rmse, r2


def main():

    print("=" * 60)
    print("AirTwinNet ST-GNN V2 Evaluation")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    data = data.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)

    train_X = []
    train_y = []
    test_X = []
    test_y = []
    test_cities = []

    print("\nCreating city-wise test sets")
    print("-" * 60)

    for city, city_data in data.groupby("city"):

        city_data = city_data.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        X_city, y_city = create_sequences(
            city_data
        )

        split_index = int(
            len(X_city) * 0.8
        )

        X_test_city = X_city[split_index:]
        y_test_city = y_city[split_index:]

        test_X.append(X_test_city)
        test_y.append(y_test_city)

        test_cities.extend(
            [city] * len(X_test_city)
        )

        print(
            f"{city}: "
            f"Test samples = {len(X_test_city)}"
        )

    test_X = np.concatenate(test_X)
    test_y = np.concatenate(test_y)

    X_test = torch.tensor(
        test_X,
        dtype=torch.float32
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    model = STGNN(
        input_features=checkpoint["input_features"],
        hidden_features=checkpoint["hidden_features"],
        output_features=checkpoint["output_features"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():

        predictions = model(
            X_test
        ).numpy()

    print("\nOverall ST-GNN V2 Results")
    print("-" * 60)

    pm25_mae, pm25_rmse, pm25_r2 = calculate_metrics(
        test_y[:, 0],
        predictions[:, 0]
    )

    pm10_mae, pm10_rmse, pm10_r2 = calculate_metrics(
        test_y[:, 1],
        predictions[:, 1]
    )

    overall_results = pd.DataFrame({
        "Target": [
            "PM2.5",
            "PM10"
        ],
        "MAE": [
            pm25_mae,
            pm10_mae
        ],
        "RMSE": [
            pm25_rmse,
            pm10_rmse
        ],
        "R2": [
            pm25_r2,
            pm10_r2
        ]
    })

    print(
        overall_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nCity-wise Results")
    print("-" * 60)

    city_results = []

    test_cities = np.asarray(test_cities)

    for city in sorted(
        np.unique(test_cities)
    ):

        mask = test_cities == city

        pm25_actual = test_y[mask, 0]
        pm25_pred = predictions[mask, 0]

        pm10_actual = test_y[mask, 1]
        pm10_pred = predictions[mask, 1]

        pm25_mae, pm25_rmse, pm25_r2 = calculate_metrics(
            pm25_actual,
            pm25_pred
        )

        pm10_mae, pm10_rmse, pm10_r2 = calculate_metrics(
            pm10_actual,
            pm10_pred
        )

        city_results.append({
            "city": city,
            "pm25_mae": pm25_mae,
            "pm25_rmse": pm25_rmse,
            "pm25_r2": pm25_r2,
            "pm10_mae": pm10_mae,
            "pm10_rmse": pm10_rmse,
            "pm10_r2": pm10_r2,
            "test_samples": int(mask.sum())
        })

        print(
            f"{city}: "
            f"PM2.5 R2={pm25_r2:.4f}, "
            f"PM10 R2={pm10_r2:.4f}"
        )

    city_results_df = pd.DataFrame(
        city_results
    )

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    overall_path = (
        "artifacts/st_gnn_v2_evaluation_results.csv"
    )

    city_path = (
        "artifacts/st_gnn_v2_city_wise_results.csv"
    )

    overall_results.to_csv(
        overall_path,
        index=False
    )

    city_results_df.to_csv(
        city_path,
        index=False
    )

    print("\nFiles saved:")
    print(overall_path)
    print(city_path)

    print("\nST-GNN V2 evaluation completed successfully.")


if __name__ == "__main__":
    main()