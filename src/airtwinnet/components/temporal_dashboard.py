import os
import pandas as pd

from src.airtwinnet.components.temporal_prediction_service import (
    TemporalPredictionService
)


DATA_PATH = "data/processed/air_quality_temporal.csv"


class TemporalDashboard:
    def __init__(
        self,
        data_path=DATA_PATH,
        model_path="artifacts/temporal_air_quality_model.pkl"
    ):
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Temporal dataset not found: {data_path}"
            )

        self.data = pd.read_csv(data_path)
        self.service = TemporalPredictionService(model_path)

    def get_forecast(self, rows=10):
        if rows <= 0:
            raise ValueError("Rows must be greater than 0.")

        sample = self.data.head(rows).copy()

        predictions = self.service.predict(
            sample[self.service.features]
        )

        result = sample[
            [
                "timestamp",
                "city",
                "pm2_5_future",
                "pm10_future"
            ]
        ].copy()

        result["predicted_pm2_5"] = predictions[
            "pm2_5_future_predicted"
        ].values

        result["predicted_pm10"] = predictions[
            "pm10_future_predicted"
        ].values

        return result


if __name__ == "__main__":
    dashboard = TemporalDashboard()

    forecast = dashboard.get_forecast(10)

    print("=" * 60)
    print("AirTwinNet Temporal Forecasting Dashboard Test")
    print("=" * 60)

    print("\nForecast generated successfully.")
    print("\nRows:", len(forecast))

    print("\nForecast:")
    print(forecast.to_string(index=False))