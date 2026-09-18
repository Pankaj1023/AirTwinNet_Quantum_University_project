import os
import pickle
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.airtwinnet.components.st_gnn import STGNN


DATA_PATH = "data/processed/air_quality_temporal.csv"
ML_MODEL_PATH = "artifacts/air_quality_model.pkl"
ST_GNN_MODEL_PATH = "artifacts/st_gnn_model_v2.pt"

ML_WEIGHT = 0.60
ST_GNN_WEIGHT = 0.40

SEQUENCE_LENGTH = 12

ST_GNN_FEATURES = [
    "temperature",
    "humidity",
    "no2",
    "co",
    "o3",
    "hour",
    "day_of_week",
    "month"
]


def create_sequences(data):

    X = []
    y = []

    features = data[ST_GNN_FEATURES].values
    targets = data[
        ["pm2_5_future", "pm10_future"]
    ].values

    for i in range(SEQUENCE_LENGTH, len(data)):

        X.append(
            features[
                i - SEQUENCE_LENGTH:i
            ]
        )

        y.append(
            targets[i]
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


def prepare_ml_features(data):

    data = data.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    data["hour"] = data["timestamp"].dt.hour

    data["day_of_week"] = (
        data["timestamp"].dt.dayofweek
    )

    data["month"] = (
        data["timestamp"].dt.month
    )

    data["day"] = (
        data["timestamp"].dt.day
    )

    data["week_of_year"] = (
        data["timestamp"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["temp_humidity"] = (
        data["temperature"]
        * data["humidity"]
    )

    data["no2_co"] = (
        data["no2"]
        * data["co"]
    )

    data["no2_o3"] = (
        data["no2"]
        * data["o3"]
    )

    data["co_o3"] = (
        data["co"]
        * data["o3"]
    )

    return data


def metrics(actual, predicted):

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
    print("AirTwinNet Hybrid Model Evaluation")
    print("=" * 60)

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    data = pd.read_csv(
        DATA_PATH
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    data = data.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # LOAD ML MODEL
    # --------------------------------------------------

    with open(
        ML_MODEL_PATH,
        "rb"
    ) as file:

        ml_package = pickle.load(file)

    pm25_ml_model = ml_package[
        "pm25_model"
    ]

    pm10_ml_model = ml_package[
        "pm10_model"
    ]

    ml_features = ml_package[
        "features"
    ]

    # --------------------------------------------------
    # LOAD ST-GNN
    # --------------------------------------------------

    checkpoint = torch.load(
        ST_GNN_MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    st_gnn = STGNN(
        input_features=checkpoint[
            "input_features"
        ],
        hidden_features=checkpoint[
            "hidden_features"
        ],
        output_features=checkpoint[
            "output_features"
        ]
    )

    st_gnn.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    st_gnn.eval()

    # --------------------------------------------------
    # CITY-WISE TEST DATA
    # --------------------------------------------------

    test_rows = []

    print("\nCreating city-wise test sets")
    print("-" * 60)

    for city, city_data in data.groupby(
        "city"
    ):

        city_data = city_data.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        X_city, y_city = create_sequences(
            city_data
        )

        split_index = int(
            len(X_city) * 0.8
        )

        X_test_city = X_city[
            split_index:
        ]

        y_test_city = y_city[
            split_index:
        ]

        timestamps = city_data[
            "timestamp"
        ].iloc[
            SEQUENCE_LENGTH + split_index:
        ].values

        cities = [
            city
        ] * len(X_test_city)

        for i in range(
            len(X_test_city)
        ):

            test_rows.append({
                "city": cities[i],
                "timestamp": timestamps[i],
                "sequence": X_test_city[i],
                "actual_pm25": y_test_city[i, 0],
                "actual_pm10": y_test_city[i, 1]
            })

        print(
            f"{city}: "
            f"Test samples = "
            f"{len(X_test_city)}"
        )

    test_df = pd.DataFrame(
        test_rows
    )

    print(
        "\nTotal test samples:",
        len(test_df)
    )

    # --------------------------------------------------
    # ST-GNN PREDICTIONS
    # --------------------------------------------------

    X_test_st = np.stack(
        test_df["sequence"].values
    )

    X_test_tensor = torch.tensor(
        X_test_st,
        dtype=torch.float32
    )

    with torch.no_grad():

        st_predictions = st_gnn(
            X_test_tensor
        ).numpy()

    test_df["st_gnn_pm25"] = (
        st_predictions[:, 0]
    )

    test_df["st_gnn_pm10"] = (
        st_predictions[:, 1]
    )

    # --------------------------------------------------
    # ML PREDICTIONS
    # --------------------------------------------------

    ml_input = test_df[
        [
            "city",
            "timestamp"
        ]
    ].copy()

    original_lookup = data[
        [
            "city",
            "timestamp",
            "temperature",
            "humidity",
            "no2",
            "co",
            "o3"
        ]
    ].copy()

    ml_input = ml_input.merge(
        original_lookup,
        on=[
            "city",
            "timestamp"
        ],
        how="left"
    )

    ml_input = prepare_ml_features(
        ml_input
    )

    missing = [
        feature
        for feature in ml_features
        if feature not in ml_input.columns
    ]

    if missing:

        raise ValueError(
            f"Missing ML features: {missing}"
        )

    X_ml = ml_input[
        ml_features
    ]

    ml_pm25 = pm25_ml_model.predict(
        X_ml
    )

    ml_pm10 = pm10_ml_model.predict(
        X_ml
    )

    test_df["ml_pm25"] = ml_pm25
    test_df["ml_pm10"] = ml_pm10

    # --------------------------------------------------
    # HYBRID PREDICTIONS
    # --------------------------------------------------

    test_df["hybrid_pm25"] = (
        ML_WEIGHT * test_df["ml_pm25"]
        +
        ST_GNN_WEIGHT * test_df["st_gnn_pm25"]
    )

    test_df["hybrid_pm10"] = (
        ML_WEIGHT * test_df["ml_pm10"]
        +
        ST_GNN_WEIGHT * test_df["st_gnn_pm10"]
    )

    # --------------------------------------------------
    # OVERALL RESULTS
    # --------------------------------------------------

    results = []

    models = [
        ("ML", "ml_pm25", "ml_pm10"),
        (
            "ST-GNN",
            "st_gnn_pm25",
            "st_gnn_pm10"
        ),
        (
            "Hybrid",
            "hybrid_pm25",
            "hybrid_pm10"
        )
    ]

    for model_name, pm25_column, pm10_column in models:

        pm25_mae, pm25_rmse, pm25_r2 = metrics(
            test_df["actual_pm25"],
            test_df[pm25_column]
        )

        pm10_mae, pm10_rmse, pm10_r2 = metrics(
            test_df["actual_pm10"],
            test_df[pm10_column]
        )

        results.append({
            "Model": model_name,
            "PM2.5_MAE": pm25_mae,
            "PM2.5_RMSE": pm25_rmse,
            "PM2.5_R2": pm25_r2,
            "PM10_MAE": pm10_mae,
            "PM10_RMSE": pm10_rmse,
            "PM10_R2": pm10_r2
        })

    results_df = pd.DataFrame(
        results
    )

    print("\nOverall Model Comparison")
    print("-" * 60)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------
    # CITY-WISE HYBRID RESULTS
    # --------------------------------------------------

    city_results = []

    for city, city_data in test_df.groupby(
        "city"
    ):

        pm25_mae, pm25_rmse, pm25_r2 = metrics(
            city_data["actual_pm25"],
            city_data["hybrid_pm25"]
        )

        pm10_mae, pm10_rmse, pm10_r2 = metrics(
            city_data["actual_pm10"],
            city_data["hybrid_pm10"]
        )

        city_results.append({
            "city": city,
            "pm25_mae": pm25_mae,
            "pm25_rmse": pm25_rmse,
            "pm25_r2": pm25_r2,
            "pm10_mae": pm10_mae,
            "pm10_rmse": pm10_rmse,
            "pm10_r2": pm10_r2,
            "test_samples": len(city_data)
        })

    city_results_df = pd.DataFrame(
        city_results
    )

    print("\nCity-wise Hybrid Results")
    print("-" * 60)

    print(
        city_results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    results_path = (
        "artifacts/hybrid_evaluation_results.csv"
    )

    city_path = (
        "artifacts/hybrid_city_wise_results.csv"
    )

    predictions_path = (
        "artifacts/hybrid_test_predictions.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    city_results_df.to_csv(
        city_path,
        index=False
    )

    test_df.drop(
        columns=["sequence"]
    ).to_csv(
        predictions_path,
        index=False
    )

    print("\nFiles saved:")
    print(results_path)
    print(city_path)
    print(predictions_path)

    print(
        "\nHybrid model evaluation "
        "completed successfully."
    )


if __name__ == "__main__":
    main()