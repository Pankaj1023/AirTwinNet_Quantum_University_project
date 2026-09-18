import os
import pickle

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.multioutput import MultiOutputRegressor


class AirQualityModelTrainer:

    def __init__(self, data_path):

        self.data_path = data_path

        self.feature_columns = [
            "temperature",
            "humidity",
            "no2",
            "co",
            "o3",
            "hour",
            "day_of_week",
            "month"
        ]

        self.target_columns = [
            "pm2_5",
            "pm10"
        ]

    def train(self):

        print(
            "Starting AirTwinNet time-aware model training..."
        )

        # Load dataset
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

        # Feature engineering
        data["hour"] = data["timestamp"].dt.hour

        data["day_of_week"] = (
            data["timestamp"].dt.dayofweek
        )

        data["month"] = (
            data["timestamp"].dt.month
        )

        # Features and targets
        X = data[
            self.feature_columns
        ]

        y = data[
            self.target_columns
        ]

        # Chronological 80/20 split
        split_index = int(
            len(data) * 0.8
        )

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        print(
            "Training samples:",
            len(X_train)
        )

        print(
            "Testing samples:",
            len(X_test)
        )

        print(
            "Training period:",
            data["timestamp"].iloc[0],
            "to",
            data["timestamp"].iloc[
                split_index - 1
            ]
        )

        print(
            "Testing period:",
            data["timestamp"].iloc[
                split_index
            ],
            "to",
            data["timestamp"].iloc[-1]
        )

        # Random Forest
        model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )

        # Train
        model.fit(
            X_train,
            y_train
        )

        # Prediction
        predictions = model.predict(
            X_test
        )

        # Evaluation
        mae = mean_absolute_error(
            y_test,
            predictions
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        rmse = mse ** 0.5

        r2 = r2_score(
            y_test,
            predictions
        )

        # Save model
        artifact_directory = "artifacts"

        os.makedirs(
            artifact_directory,
            exist_ok=True
        )

        model_path = os.path.join(
            artifact_directory,
            "air_quality_model.pkl"
        )

        with open(
            model_path,
            "wb"
        ) as file:

            pickle.dump(
                model,
                file
            )

        print(
            "Trained model saved successfully."
        )

        print(
            "Model path:",
            model_path
        )

        print(
            "Model training completed successfully."
        )

        print(
            "MAE:",
            mae
        )

        print(
            "RMSE:",
            rmse
        )

        print(
            "R2 Score:",
            r2
        )

        return model


if __name__ == "__main__":

    trainer = AirQualityModelTrainer(
        "data/raw/air_quality_sample.csv"
    )

    trainer.train()