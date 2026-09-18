import os
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.airtwinnet.components.st_gnn import STGNN
import torch


DATA_PATH = "data/processed/air_quality_temporal.csv"
TEMPORAL_MODEL_PATH = "artifacts/temporal_air_quality_model.pkl"
ST_GNN_MODEL_PATH = "artifacts/st_gnn_model_v2.pt"

OUTPUT_DIR = "artifacts"


SEQ_LEN = 12
ML_WEIGHT = 0.60
ST_GNN_WEIGHT = 0.40


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    return mae, rmse, r2


def create_ml_features(df):
    data = df.copy()

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["month"] = data["timestamp"].dt.month
    data["day"] = data["timestamp"].dt.day
    data["week_of_year"] = data["timestamp"].dt.isocalendar().week.astype(int)

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["temp_humidity"] = (
        data["temperature"] * data["humidity"]
    )

    data["no2_co"] = (
        data["no2"] * data["co"]
    )

    data["no2_o3"] = (
        data["no2"] * data["o3"]
    )

    data["co_o3"] = (
        data["co"] * data["o3"]
    )

    return data


def load_temporal_model():
    if not os.path.exists(TEMPORAL_MODEL_PATH):
        raise FileNotFoundError(
            f"Temporal model not found: {TEMPORAL_MODEL_PATH}"
        )

    with open(TEMPORAL_MODEL_PATH, "rb") as file:
        package = pickle.load(file)

    return package


def load_st_gnn():
    if not os.path.exists(ST_GNN_MODEL_PATH):
        raise FileNotFoundError(
            f"ST-GNN model not found: {ST_GNN_MODEL_PATH}"
        )

    package = torch.load(
        ST_GNN_MODEL_PATH,
        map_location="cpu"
    )

    model = STGNN(
        input_features=package["input_features"],
        hidden_features=package["hidden_features"],
        output_features=package["output_features"]
    )

    model.load_state_dict(
        package["model_state_dict"]
    )

    model.eval()

    return model

def create_sequences(city_df):
    features = [
        "temperature",
        "humidity",
        "no2",
        "co",
        "o3",
        "hour",
        "day_of_week",
        "month"
    ]

    X = city_df[features].values.astype(np.float32)

    sequences = []
    target_indices = []

    for i in range(SEQ_LEN, len(city_df)):
        sequence = X[i - SEQ_LEN:i]

        sequences.append(sequence)
        target_indices.append(i)

    return (
        np.array(sequences, dtype=np.float32),
        target_indices
    )


def main():

    print("=" * 60)
    print("AirTwinNet Proper Future Hybrid Evaluation")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)

    print()
    print("Dataset loaded")
    print("-" * 60)
    print(f"Total rows: {len(df)}")
    print(f"Cities: {df['city'].nunique()}")

    temporal_package = load_temporal_model()

    temporal_pm25_model = temporal_package["pm25_model"]
    temporal_pm10_model = temporal_package["pm10_model"]
    temporal_features = temporal_package["features"]

    st_gnn_model = load_st_gnn()

    all_results = []

    print()
    print("Creating city-wise chronological test sets")
    print("-" * 60)

    for city in sorted(df["city"].unique()):

        city_df = df[
            df["city"] == city
        ].copy()

        city_df = city_df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        split_index = int(
            len(city_df) * 0.80
        )

        test_df = city_df.iloc[
            split_index:
        ].copy()

        print(
            f"{city}: Test samples = {len(test_df)}"
        )

        # -------------------------------------------------
        # ML TEMPORAL MODEL
        # -------------------------------------------------

        ml_test = create_ml_features(test_df)

        X_ml = ml_test[
            temporal_features
        ]

        ml_pm25 = temporal_pm25_model.predict(
            X_ml
        )

        ml_pm10 = temporal_pm10_model.predict(
            X_ml
        )

        # -------------------------------------------------
        # ST-GNN
        # -------------------------------------------------

        sequence_source = city_df.iloc[
            split_index - SEQ_LEN:
        ].copy()

        X_seq, target_indices = create_sequences(
            sequence_source
        )

        # Only retain sequences whose target
        # belongs to the actual test period.

        valid_sequences = []

        for seq, local_index in zip(
            X_seq,
            target_indices
        ):

            global_timestamp = sequence_source.iloc[
                local_index
            ]["timestamp"]

            if global_timestamp >= test_df.iloc[0]["timestamp"]:
                valid_sequences.append(seq)

        X_seq = np.array(
            valid_sequences,
            dtype=np.float32
        )

        with torch.no_grad():

            tensor_input = torch.tensor(
                X_seq,
                dtype=torch.float32
            )

            st_output = st_gnn_model(
                tensor_input
            ).numpy()

        st_pm25 = st_output[:, 0]
        st_pm10 = st_output[:, 1]

        # -------------------------------------------------
        # ALIGN LENGTHS
        # -------------------------------------------------

        min_length = min(
            len(test_df),
            len(ml_pm25),
            len(st_pm25)
        )

        actual_df = test_df.iloc[
            :min_length
        ].copy()

        ml_pm25 = ml_pm25[:min_length]
        ml_pm10 = ml_pm10[:min_length]

        st_pm25 = st_pm25[:min_length]
        st_pm10 = st_pm10[:min_length]

        # -------------------------------------------------
        # HYBRID
        # -------------------------------------------------

        hybrid_pm25 = (
            ML_WEIGHT * ml_pm25
            + ST_GNN_WEIGHT * st_pm25
        )

        hybrid_pm10 = (
            ML_WEIGHT * ml_pm10
            + ST_GNN_WEIGHT * st_pm10
        )

        # -------------------------------------------------
        # SAVE PREDICTIONS
        # -------------------------------------------------

        city_results = pd.DataFrame({

            "city": city,

            "timestamp": actual_df[
                "timestamp"
            ].values,

            "actual_pm2_5": actual_df[
                "pm2_5_future"
            ].values,

            "ml_pm2_5": ml_pm25,

            "st_gnn_pm2_5": st_pm25,

            "hybrid_pm2_5": hybrid_pm25,

            "actual_pm10": actual_df[
                "pm10_future"
            ].values,

            "ml_pm10": ml_pm10,

            "st_gnn_pm10": st_pm10,

            "hybrid_pm10": hybrid_pm10
        })

        all_results.append(
            city_results
        )

    # -----------------------------------------------------
    # COMBINE RESULTS
    # -----------------------------------------------------

    results = pd.concat(
        all_results,
        ignore_index=True
    )

    print()
    print("=" * 60)
    print("Overall Future Forecast Comparison")
    print("=" * 60)

    comparison = []

    models = {
        "Temporal ML": (
            "ml_pm2_5",
            "ml_pm10"
        ),

        "ST-GNN": (
            "st_gnn_pm2_5",
            "st_gnn_pm10"
        ),

        "Hybrid": (
            "hybrid_pm2_5",
            "hybrid_pm10"
        )
    }

    for model_name, columns in models.items():

        pm25_mae, pm25_rmse, pm25_r2 = calculate_metrics(
            results["actual_pm2_5"],
            results[columns[0]]
        )

        pm10_mae, pm10_rmse, pm10_r2 = calculate_metrics(
            results["actual_pm10"],
            results[columns[1]]
        )

        comparison.append({

            "Model": model_name,

            "PM2.5_MAE": round(
                pm25_mae, 4
            ),

            "PM2.5_RMSE": round(
                pm25_rmse, 4
            ),

            "PM2.5_R2": round(
                pm25_r2, 4
            ),

            "PM10_MAE": round(
                pm10_mae, 4
            ),

            "PM10_RMSE": round(
                pm10_rmse, 4
            ),

            "PM10_R2": round(
                pm10_r2, 4
            )
        })

    comparison_df = pd.DataFrame(
        comparison
    )

    print(comparison_df.to_string(
        index=False
    ))

    # -----------------------------------------------------
    # CITY-WISE HYBRID
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("City-wise Future Hybrid Results")
    print("=" * 60)

    city_results = []

    for city in sorted(
        results["city"].unique()
    ):

        city_data = results[
            results["city"] == city
        ]

        pm25_mae, pm25_rmse, pm25_r2 = calculate_metrics(
            city_data["actual_pm2_5"],
            city_data["hybrid_pm2_5"]
        )

        pm10_mae, pm10_rmse, pm10_r2 = calculate_metrics(
            city_data["actual_pm10"],
            city_data["hybrid_pm10"]
        )

        city_results.append({

            "city": city,

            "pm25_mae": round(
                pm25_mae, 4
            ),

            "pm25_rmse": round(
                pm25_rmse, 4
            ),

            "pm25_r2": round(
                pm25_r2, 4
            ),

            "pm10_mae": round(
                pm10_mae, 4
            ),

            "pm10_rmse": round(
                pm10_rmse, 4
            ),

            "pm10_r2": round(
                pm10_r2, 4
            ),

            "test_samples": len(
                city_data
            )
        })

    city_df = pd.DataFrame(
        city_results
    )

    print(city_df.to_string(
        index=False
    ))

    # -----------------------------------------------------
    # SAVE FILES
    # -----------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    comparison_path = (
        f"{OUTPUT_DIR}/"
        "hybrid_future_evaluation_results.csv"
    )

    city_path = (
        f"{OUTPUT_DIR}/"
        "hybrid_future_city_wise_results.csv"
    )

    predictions_path = (
        f"{OUTPUT_DIR}/"
        "hybrid_future_test_predictions.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    city_df.to_csv(
        city_path,
        index=False
    )

    results.to_csv(
        predictions_path,
        index=False
    )

    print()
    print("=" * 60)
    print("Files saved:")
    print(comparison_path)
    print(city_path)
    print(predictions_path)
    print("=" * 60)

    print()
    print(
        "Proper future hybrid evaluation completed successfully."
    )


if __name__ == "__main__":
    main()