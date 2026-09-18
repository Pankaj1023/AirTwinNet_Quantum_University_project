import os
import pickle
import numpy as np
import pandas as pd
import torch

from src.airtwinnet.components.st_gnn import STGNN


ML_MODEL_PATH = "artifacts/air_quality_model.pkl"
ST_GNN_MODEL_PATH = "artifacts/st_gnn_model_v2.pt"
DATA_PATH = "data/processed/air_quality_temporal.csv"

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


class HybridPredictionEngine:

    def __init__(
        self,
        ml_model_path=ML_MODEL_PATH,
        st_gnn_model_path=ST_GNN_MODEL_PATH
    ):

        if not os.path.exists(ml_model_path):
            raise FileNotFoundError(
                f"ML model not found: {ml_model_path}"
            )

        if not os.path.exists(st_gnn_model_path):
            raise FileNotFoundError(
                f"ST-GNN model not found: {st_gnn_model_path}"
            )

        with open(ml_model_path, "rb") as file:
            ml_package = pickle.load(file)

        self.pm25_ml_model = ml_package["pm25_model"]
        self.pm10_ml_model = ml_package["pm10_model"]
        self.ml_features = ml_package["features"]

        checkpoint = torch.load(
            st_gnn_model_path,
            map_location="cpu",
            weights_only=False
        )

        self.st_gnn = STGNN(
            input_features=checkpoint["input_features"],
            hidden_features=checkpoint["hidden_features"],
            output_features=checkpoint["output_features"]
        )

        self.st_gnn.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.st_gnn.eval()

        self.st_gnn_features = checkpoint["features"]

    def prepare_ml_features(self, data):

        data = data.copy()

        # Calendar features
        if "timestamp" in data.columns:

            data["timestamp"] = pd.to_datetime(
                data["timestamp"]
            )

            data["hour"] = data["timestamp"].dt.hour
            data["day_of_week"] = (
                data["timestamp"].dt.dayofweek
            )
            data["month"] = data["timestamp"].dt.month
            data["day"] = data["timestamp"].dt.day
            data["week_of_year"] = (
                data["timestamp"].dt.isocalendar().week.astype(int)
            )

        # Weekend
        data["is_weekend"] = (
            data["day_of_week"] >= 5
        ).astype(int)

        # Interaction features
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

    def predict_ml(self, data):

        data = self.prepare_ml_features(data)

        missing_features = [
            feature
            for feature in self.ml_features
            if feature not in data.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing ML features: {missing_features}"
            )

        X = data[self.ml_features]

        pm25 = self.pm25_ml_model.predict(X)
        pm10 = self.pm10_ml_model.predict(X)

        return pm25, pm10

    def predict_st_gnn(self, sequence):

        sequence_tensor = torch.tensor(
            sequence,
            dtype=torch.float32
        )

        if sequence_tensor.ndim == 2:
            sequence_tensor = (
                sequence_tensor.unsqueeze(0)
            )

        with torch.no_grad():

            prediction = self.st_gnn(
                sequence_tensor
            ).numpy()

        return (
            prediction[:, 0],
            prediction[:, 1]
        )

    def predict(self, data, sequence):

        if isinstance(data, dict):

            data = pd.DataFrame([data])

        elif not isinstance(data, pd.DataFrame):

            raise TypeError(
                "Data must be a DataFrame or dictionary."
            )

        ml_pm25, ml_pm10 = self.predict_ml(
            data
        )

        st_pm25, st_pm10 = self.predict_st_gnn(
            sequence
        )

        hybrid_pm25 = (
            ML_WEIGHT * ml_pm25[0]
            + ST_GNN_WEIGHT * st_pm25[0]
        )

        hybrid_pm10 = (
            ML_WEIGHT * ml_pm10[0]
            + ST_GNN_WEIGHT * st_pm10[0]
        )

        return {
            "ml_pm2_5": float(ml_pm25[0]),
            "ml_pm10": float(ml_pm10[0]),
            "st_gnn_pm2_5": float(st_pm25[0]),
            "st_gnn_pm10": float(st_pm10[0]),
            "hybrid_pm2_5": float(hybrid_pm25),
            "hybrid_pm10": float(hybrid_pm10),
            "ml_weight": ML_WEIGHT,
            "st_gnn_weight": ST_GNN_WEIGHT
        }


def main():

    print("=" * 60)
    print("AirTwinNet Hybrid Prediction Engine")
    print("=" * 60)

    data = pd.read_csv(DATA_PATH)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    data = data.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)

    city = data["city"].iloc[0]

    city_data = data[
        data["city"] == city
    ].reset_index(drop=True)

    sample = city_data.iloc[
        SEQUENCE_LENGTH:
        SEQUENCE_LENGTH + 1
    ].copy()

    sequence = city_data[
        ST_GNN_FEATURES
    ].iloc[
        :SEQUENCE_LENGTH
    ].values.astype(np.float32)

    engine = HybridPredictionEngine()

    result = engine.predict(
        sample,
        sequence
    )

    print("\nCity:")
    print(city)

    print("\nPrediction Results")
    print("-" * 60)

    print(
        f"ML PM2.5       : "
        f"{result['ml_pm2_5']:.4f}"
    )

    print(
        f"ST-GNN PM2.5   : "
        f"{result['st_gnn_pm2_5']:.4f}"
    )

    print(
        f"Hybrid PM2.5   : "
        f"{result['hybrid_pm2_5']:.4f}"
    )

    print()

    print(
        f"ML PM10        : "
        f"{result['ml_pm10']:.4f}"
    )

    print(
        f"ST-GNN PM10    : "
        f"{result['st_gnn_pm10']:.4f}"
    )

    print(
        f"Hybrid PM10    : "
        f"{result['hybrid_pm10']:.4f}"
    )

    print("\nWeights")
    print("-" * 60)

    print(
        f"ML weight      : "
        f"{ML_WEIGHT}"
    )

    print(
        f"ST-GNN weight  : "
        f"{ST_GNN_WEIGHT}"
    )

    print(
        "\nHybrid prediction engine "
        "test completed successfully."
    )


if __name__ == "__main__":
    main()