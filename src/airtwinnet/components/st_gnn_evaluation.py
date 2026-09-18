import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.airtwinnet.components.st_gnn import STGNN


DATA_PATH = "data/processed/air_quality_temporal.csv"
MODEL_PATH = "artifacts/st_gnn_model.pt"

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

    values = data[FEATURES].values
    targets = data[TARGETS].values

    for i in range(SEQUENCE_LENGTH, len(data)):
        X.append(values[i - SEQUENCE_LENGTH:i])
        y.append(targets[i])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


def main():

    print("=" * 60)
    print("AirTwinNet ST-GNN Evaluation")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"ST-GNN model not found: {MODEL_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    data = data.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)

    X_all = []
    y_all = []

    for city, city_data in data.groupby("city"):

        city_data = city_data.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        X_city, y_city = create_sequences(
            city_data
        )

        X_all.append(X_city)
        y_all.append(y_city)

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)

    split_index = int(len(X) * 0.8)

    X_test = X[split_index:]
    y_test = y[split_index:]

    X_test = torch.tensor(X_test)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    model = STGNN(
        input_features=len(FEATURES),
        hidden_features=checkpoint["hidden_features"],
        output_features=2
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():

        predictions = model(
            X_test
        ).numpy()

    pm25_actual = y_test[:, 0]
    pm25_predicted = predictions[:, 0]

    pm10_actual = y_test[:, 1]
    pm10_predicted = predictions[:, 1]

    pm25_mae = mean_absolute_error(
        pm25_actual,
        pm25_predicted
    )

    pm25_rmse = np.sqrt(
        mean_squared_error(
            pm25_actual,
            pm25_predicted
        )
    )

    pm25_r2 = r2_score(
        pm25_actual,
        pm25_predicted
    )

    pm10_mae = mean_absolute_error(
        pm10_actual,
        pm10_predicted
    )

    pm10_rmse = np.sqrt(
        mean_squared_error(
            pm10_actual,
            pm10_predicted
        )
    )

    pm10_r2 = r2_score(
        pm10_actual,
        pm10_predicted
    )

    results = pd.DataFrame({
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

    print("\nST-GNN Test Results")
    print("-" * 60)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nTest samples:", len(X_test))

    output_path = (
        "artifacts/st_gnn_evaluation_results.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nResults saved to: {output_path}"
    )

    print("\nST-GNN evaluation completed successfully.")


if __name__ == "__main__":
    main()