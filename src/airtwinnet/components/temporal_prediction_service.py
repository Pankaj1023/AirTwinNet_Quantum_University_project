import os
import pickle
import pandas as pd


MODEL_PATH = "artifacts/temporal_air_quality_model.pkl"


class TemporalPredictionService:

    def __init__(self, model_path=MODEL_PATH):

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Temporal model not found: {model_path}"
            )

        with open(model_path, "rb") as file:
            package = pickle.load(file)

        self.pm25_model = package["pm25_model"]
        self.pm10_model = package["pm10_model"]
        self.features = package["features"]

    def predict(self, data):

        if isinstance(data, dict):
            data = pd.DataFrame([data])

        elif isinstance(data, pd.Series):
            data = data.to_frame().T

        elif not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input must be a dictionary, Series, or DataFrame."
            )

        missing_features = [
            feature
            for feature in self.features
            if feature not in data.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}"
            )

        X = data[self.features]

        pm25_prediction = self.pm25_model.predict(X)
        pm10_prediction = self.pm10_model.predict(X)

        result = data.copy()

        result["pm2_5_future_predicted"] = pm25_prediction
        result["pm10_future_predicted"] = pm10_prediction

        return result


if __name__ == "__main__":

    print("=" * 60)
    print("AirTwinNet Temporal Prediction Service Test")
    print("=" * 60)

    service = TemporalPredictionService()

    print("\nModel loaded successfully.")

    print("\nRequired features:")
    print(service.features)

    print("\nNumber of features:")
    print(len(service.features))

    print("\nTemporal prediction service is ready.")