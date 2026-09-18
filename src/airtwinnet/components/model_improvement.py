import os
import sys
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

from src.airtwinnet.exception.exception import AirTwinNetException
from src.airtwinnet.logger.logger import logger


class ModelImprovement:

    def __init__(self):

        self.data_path = "data/raw/air_quality_sample.csv"
        self.model_path = "artifacts/air_quality_model.pkl"

    # ==========================================================
    # LOAD DATA
    # ==========================================================

    def load_data(self):

        try:

            df = pd.read_csv(self.data_path)

            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce"
            )

            df = df.dropna(
                subset=["timestamp"]
            )

            df = df.sort_values(
                "timestamp"
            ).reset_index(drop=True)

            return df

        except Exception as e:

            logger.exception(
                "Error while loading data"
            )

            raise AirTwinNetException(
                e,
                sys
            ) from e

    # ==========================================================
    # FEATURE ENGINEERING
    # ==========================================================

    def create_features(self, df):

        try:

            df = df.copy()

            # --------------------------------------------------
            # Time Features
            # --------------------------------------------------

            df["hour"] = df["timestamp"].dt.hour

            df["day_of_week"] = (
                df["timestamp"].dt.dayofweek
            )

            df["month"] = (
                df["timestamp"].dt.month
            )

            df["day"] = (
                df["timestamp"].dt.day
            )

            df["week_of_year"] = (
                df["timestamp"]
                .dt.isocalendar()
                .week
                .astype(int)
            )

            df["is_weekend"] = (
                df["day_of_week"] >= 5
            ).astype(int)

            # --------------------------------------------------
            # Interaction Features
            # --------------------------------------------------

            df["temp_humidity"] = (
                df["temperature"] *
                df["humidity"]
            )

            df["no2_co"] = (
                df["no2"] *
                df["co"]
            )

            df["no2_o3"] = (
                df["no2"] *
                df["o3"]
            )

            df["co_o3"] = (
                df["co"] *
                df["o3"]
            )

            # --------------------------------------------------
            # Final Feature List
            # --------------------------------------------------

            feature_columns = [

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

            X = df[feature_columns]

            y_pm25 = df["pm2_5"]

            y_pm10 = df["pm10"]

            return (
                X,
                y_pm25,
                y_pm10
            )

        except Exception as e:

            logger.exception(
                "Error during feature engineering"
            )

            raise AirTwinNetException(
                e,
                sys
            ) from e

    # ==========================================================
    # MODEL FACTORY
    # ==========================================================

    def create_models(self):

        models = {

            "RandomForest": RandomForestRegressor(
                n_estimators=150,
                random_state=42,
                n_jobs=-1
            ),

            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=150,
                random_state=42,
                n_jobs=-1
            ),

            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            ),

            "XGBoost": XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )
        }

        return models

    # ==========================================================
    # MODEL EVALUATION
    # ==========================================================

    def evaluate_model(
        self,
        model,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        return {
            "model": model,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

    # ==========================================================
    # RUN MODEL IMPROVEMENT
    # ==========================================================

    def run(self):

        try:

            print(
                "\nStarting AirTwinNet model improvement...\n"
            )

            # --------------------------------------------------
            # Load Data
            # --------------------------------------------------

            df = self.load_data()

            print(
                f"Total dataset rows: {len(df)}"
            )

            # --------------------------------------------------
            # Create Features
            # --------------------------------------------------

            (
                X,
                y_pm25,
                y_pm10
            ) = self.create_features(df)

            # --------------------------------------------------
            # Chronological 80/20 Split
            # --------------------------------------------------

            split_index = int(
                len(X) * 0.80
            )

            X_train = X.iloc[
                :split_index
            ]

            X_test = X.iloc[
                split_index:
            ]

            y_pm25_train = y_pm25.iloc[
                :split_index
            ]

            y_pm25_test = y_pm25.iloc[
                split_index:
            ]

            y_pm10_train = y_pm10.iloc[
                :split_index
            ]

            y_pm10_test = y_pm10.iloc[
                split_index:
            ]

            print(
                f"Training samples: {len(X_train)}"
            )

            print(
                f"Testing samples: {len(X_test)}"
            )

            # --------------------------------------------------
            # Model Definitions
            # --------------------------------------------------

            models = self.create_models()

            # ==================================================
            # PM2.5 MODEL COMPARISON
            # ==================================================

            print(
                "\n================ PM2.5 MODEL COMPARISON ================\n"
            )

            pm25_results = {}

            for name, model in models.items():

                print(
                    f"Training {name} for PM2.5..."
                )

                result = self.evaluate_model(
                    model,
                    X_train,
                    X_test,
                    y_pm25_train,
                    y_pm25_test
                )

                pm25_results[name] = result

                print(
                    f"{name}: "
                    f"MAE={result['MAE']:.4f}, "
                    f"RMSE={result['RMSE']:.4f}, "
                    f"R2={result['R2']:.4f}"
                )

            # ==================================================
            # PM10 MODEL COMPARISON
            # ==================================================

            print(
                "\n================ PM10 MODEL COMPARISON ================\n"
            )

            pm10_results = {}

            for name in models.keys():

                print(
                    f"Training {name} for PM10..."
                )

                # Fresh model
                fresh_models = self.create_models()

                fresh_model = fresh_models[
                    name
                ]

                result = self.evaluate_model(
                    fresh_model,
                    X_train,
                    X_test,
                    y_pm10_train,
                    y_pm10_test
                )

                pm10_results[name] = result

                print(
                    f"{name}: "
                    f"MAE={result['MAE']:.4f}, "
                    f"RMSE={result['RMSE']:.4f}, "
                    f"R2={result['R2']:.4f}"
                )

            # ==================================================
            # SELECT BEST MODELS
            # ==================================================

            best_pm25_name = max(
                pm25_results,
                key=lambda name:
                pm25_results[name]["R2"]
            )

            best_pm10_name = max(
                pm10_results,
                key=lambda name:
                pm10_results[name]["R2"]
            )

            best_pm25_model = pm25_results[
                best_pm25_name
            ]["model"]

            best_pm10_model = pm10_results[
                best_pm10_name
            ]["model"]

            # ==================================================
            # SAVE IMPROVED MODEL
            # ==================================================

            os.makedirs(
                "artifacts",
                exist_ok=True
            )

            improved_model = {

                "pm25_model": best_pm25_model,

                "pm10_model": best_pm10_model,

                "features": list(
                    X.columns
                ),

                "pm25_model_name":
                    best_pm25_name,

                "pm10_model_name":
                    best_pm10_name
            }

            with open(
                self.model_path,
                "wb"
            ) as file:

                pickle.dump(
                    improved_model,
                    file
                )

            # ==================================================
            # FINAL RESULTS
            # ==================================================

            print(
                "\n================================================"
            )

            print(
                "BEST MODEL RESULTS"
            )

            print(
                "================================================"
            )

            print(
                f"PM2.5 Best Model: "
                f"{best_pm25_name}"
            )

            print(
                f"PM2.5 MAE: "
                f"{pm25_results[best_pm25_name]['MAE']:.4f}"
            )

            print(
                f"PM2.5 RMSE: "
                f"{pm25_results[best_pm25_name]['RMSE']:.4f}"
            )

            print(
                f"PM2.5 R2: "
                f"{pm25_results[best_pm25_name]['R2']:.4f}"
            )

            print()

            print(
                f"PM10 Best Model: "
                f"{best_pm10_name}"
            )

            print(
                f"PM10 MAE: "
                f"{pm10_results[best_pm10_name]['MAE']:.4f}"
            )

            print(
                f"PM10 RMSE: "
                f"{pm10_results[best_pm10_name]['RMSE']:.4f}"
            )

            print(
                f"PM10 R2: "
                f"{pm10_results[best_pm10_name]['R2']:.4f}"
            )

            print(
                "\nImproved model saved successfully:"
            )

            print(
                self.model_path
            )

        except Exception as e:

            logger.exception(
                "Error during model improvement"
            )

            raise AirTwinNetException(
                e,
                sys
            ) from e


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":

    improvement = ModelImprovement()

    improvement.run()