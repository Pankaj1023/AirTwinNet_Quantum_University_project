import pickle
import pandas as pd


class AirQualityModelPrediction:

    def __init__(self, model_path):

        self.model_path = model_path

        with open(self.model_path, "rb") as file:
            self.model = pickle.load(file)

    def predict(self, input_data):

        features = [
            "temperature",
            "humidity",
            "no2",
            "co",
            "o3"
        ]

        X = input_data[features]

        predictions = self.model.predict(X)

        result = pd.DataFrame(
            predictions,
            columns=["pm2_5_prediction", "pm10_prediction"]
        )

        return result


if __name__ == "__main__":

    model = AirQualityModelPrediction(
        "artifacts/air_quality_model.pkl"
    )

    data = pd.read_csv(
        "data/raw/air_quality_sample.csv"
    )

    sample = data.tail(1)

    prediction = model.predict(sample)

    print("\nLatest Input:")
    print(
        sample[
            [
                "timestamp",
                "city",
                "temperature",
                "humidity",
                "no2",
                "co",
                "o3"
            ]
        ].to_string(index=False)
    )

    print("\nActual Values:")
    print(
        sample[
            ["pm2_5", "pm10"]
        ].to_string(index=False)
    )

    print("\nModel Prediction:")
    print(prediction.to_string(index=False))