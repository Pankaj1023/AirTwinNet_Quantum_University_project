import os
import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class AirQualityModelTrainer:

    def __init__(self, data_path):

        self.data_path = data_path

        self.feature_columns = [
            "temperature",
            "humidity",
            "no2",
            "co",
            "o3"
        ]

        self.target_columns = [
            "pm2_5",
            "pm10"
        ]

    def train(self):

        print("Starting AirTwinNet model training...")

        # Load dataset
        data = pd.read_csv(self.data_path)

        # Features and targets
        X = data[self.feature_columns]
        y = data[self.target_columns]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        print("Training samples:", len(X_train))
        print("Testing samples:", len(X_test))

        # Multi-output Random Forest
        model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

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

        print("Model training completed successfully.")
        print("MAE:", mae)
        print("RMSE:", rmse)
        print("R2 Score:", r2)

        return model


if __name__ == "__main__":

    trainer = AirQualityModelTrainer(
        "data/raw/air_quality_sample.csv"
    )

    trainer.train()