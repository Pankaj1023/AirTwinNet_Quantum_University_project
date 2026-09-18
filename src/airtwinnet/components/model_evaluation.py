import os
import pickle

import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class AirQualityModelEvaluator:

    def __init__(
        self,
        data_path,
        model_path="artifacts/air_quality_model.pkl"
    ):

        self.data_path = data_path
        self.model_path = model_path

        self.feature_columns = [
            "temperature",
            "humidity",
            "no2",
            "co",
            "o3",
            "hour",
            "day_of_week",
            "month",
            "day",
            "week_of_year",
            "is_weekend",
            "temp_humidity",
            "no2_co",
            "no2_o3",
            "co_o3"
        ]

        self.target_columns = [
            "pm2_5",
            "pm10"
        ]

    def load_model(self):

        with open(
            self.model_path,
            "rb"
        ) as file:

            model = pickle.load(file)

        return model

    def prepare_data(self):

        data = pd.read_csv(
            self.data_path
        )

        # Convert timestamp
        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce"
        )

        # Remove invalid timestamps
        data = data.dropna(
            subset=["timestamp"]
        )

        # Sort chronologically
        data = data.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        # --------------------------------------------------
        # Feature Engineering
        # --------------------------------------------------

        data["hour"] = (
            data["timestamp"].dt.hour
        )

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

        # Interaction features

        data["temp_humidity"] = (
            data["temperature"] *
            data["humidity"]
        )

        data["no2_co"] = (
            data["no2"] *
            data["co"]
        )

        data["no2_o3"] = (
            data["no2"] *
            data["o3"]
        )

        data["co_o3"] = (
            data["co"] *
            data["o3"]
        )

        return data

    def evaluate(self):

        print(
            "Starting AirTwinNet detailed model evaluation..."
        )

        # --------------------------------------------------
        # Load Data
        # --------------------------------------------------

        data = self.prepare_data()

        # --------------------------------------------------
        # Load Model Package
        # --------------------------------------------------

        model_package = self.load_model()

        pm25_model = model_package[
            "pm25_model"
        ]

        pm10_model = model_package[
            "pm10_model"
        ]

        # Use exact feature order saved with model
        self.feature_columns = model_package[
            "features"
        ]

        # --------------------------------------------------
        # Chronological 80/20 Split
        # --------------------------------------------------

        split_index = int(
            len(data) * 0.8
        )

        test_data = data.iloc[
            split_index:
        ].copy()

        X_test = test_data[
            self.feature_columns
        ]

        y_test = test_data[
            self.target_columns
        ]

        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------

        print(
            "\nGenerating PM2.5 predictions..."
        )

        pm25_predictions = pm25_model.predict(
            X_test
        )

        print(
            "Generating PM10 predictions..."
        )

        pm10_predictions = pm10_model.predict(
            X_test
        )

        predictions = pd.DataFrame(
            {
                "pm2_5": pm25_predictions,
                "pm10": pm10_predictions
            },
            index=test_data.index
        )

        # --------------------------------------------------
        # Add Predictions to Test Data
        # --------------------------------------------------

        test_data["predicted_pm2_5"] = (
            pm25_predictions
        )

        test_data["predicted_pm10"] = (
            pm10_predictions
        )

        # --------------------------------------------------
        # Overall Metrics
        # --------------------------------------------------

        overall_mae = mean_absolute_error(
            y_test,
            predictions
        )

        overall_rmse = mean_squared_error(
            y_test,
            predictions
        ) ** 0.5

        overall_r2 = r2_score(
            y_test,
            predictions
        )

        print(
            "\nOverall Performance:"
        )

        print(
            "MAE:",
            overall_mae
        )

        print(
            "RMSE:",
            overall_rmse
        )

        print(
            "R2 Score:",
            overall_r2
        )

        # --------------------------------------------------
        # PM2.5 Metrics
        # --------------------------------------------------

        pm25_mae = mean_absolute_error(
            y_test["pm2_5"],
            pm25_predictions
        )

        pm25_rmse = mean_squared_error(
            y_test["pm2_5"],
            pm25_predictions
        ) ** 0.5

        pm25_r2 = r2_score(
            y_test["pm2_5"],
            pm25_predictions
        )

        print(
            "\nPM2.5 Performance:"
        )

        print(
            "MAE:",
            pm25_mae
        )

        print(
            "RMSE:",
            pm25_rmse
        )

        print(
            "R2 Score:",
            pm25_r2
        )

        # --------------------------------------------------
        # PM10 Metrics
        # --------------------------------------------------

        pm10_mae = mean_absolute_error(
            y_test["pm10"],
            pm10_predictions
        )

        pm10_rmse = mean_squared_error(
            y_test["pm10"],
            pm10_predictions
        ) ** 0.5

        pm10_r2 = r2_score(
            y_test["pm10"],
            pm10_predictions
        )

        print(
            "\nPM10 Performance:"
        )

        print(
            "MAE:",
            pm10_mae
        )

        print(
            "RMSE:",
            pm10_rmse
        )

        print(
            "R2 Score:",
            pm10_r2
        )

        # --------------------------------------------------
        # City-wise Evaluation
        # --------------------------------------------------

        print(
            "\nCity-wise Performance:"
        )

        city_results = []

        for city in sorted(
            test_data["city"].unique()
        ):

            city_data = test_data[
                test_data["city"] == city
            ]

            actual_pm25 = city_data[
                "pm2_5"
            ]

            predicted_pm25 = city_data[
                "predicted_pm2_5"
            ]

            actual_pm10 = city_data[
                "pm10"
            ]

            predicted_pm10 = city_data[
                "predicted_pm10"
            ]

            city_pm25_r2 = r2_score(
                actual_pm25,
                predicted_pm25
            )

            city_pm10_r2 = r2_score(
                actual_pm10,
                predicted_pm10
            )

            city_results.append(
                {
                    "city": city,

                    "pm2_5_mae":
                        mean_absolute_error(
                            actual_pm25,
                            predicted_pm25
                        ),

                    "pm2_5_rmse":
                        mean_squared_error(
                            actual_pm25,
                            predicted_pm25
                        ) ** 0.5,

                    "pm2_5_r2":
                        city_pm25_r2,

                    "pm10_mae":
                        mean_absolute_error(
                            actual_pm10,
                            predicted_pm10
                        ),

                    "pm10_rmse":
                        mean_squared_error(
                            actual_pm10,
                            predicted_pm10
                        ) ** 0.5,

                    "pm10_r2":
                        city_pm10_r2
                }
            )

            print(
                f"\n{city}:"
            )

            print(
                "PM2.5 R2:",
                city_pm25_r2
            )

            print(
                "PM10 R2:",
                city_pm10_r2
            )

        # --------------------------------------------------
        # Save Evaluation Artifacts
        # --------------------------------------------------

        artifact_directory = "artifacts"

        os.makedirs(
            artifact_directory,
            exist_ok=True
        )

        # Save predictions

        predictions_path = os.path.join(
            artifact_directory,
            "model_predictions.csv"
        )

        test_data.to_csv(
            predictions_path,
            index=False
        )

        # Save city-wise evaluation

        city_results_path = os.path.join(
            artifact_directory,
            "city_wise_evaluation.csv"
        )

        pd.DataFrame(
            city_results
        ).to_csv(
            city_results_path,
            index=False
        )

        print(
            "\nEvaluation files saved successfully."
        )

        print(
            "Predictions:",
            predictions_path
        )

        print(
            "City-wise evaluation:",
            city_results_path
        )


if __name__ == "__main__":

    evaluator = AirQualityModelEvaluator(
        "data/raw/air_quality_sample.csv"
    )

    evaluator.evaluate()