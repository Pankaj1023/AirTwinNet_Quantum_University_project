import os
import pickle
import pandas as pd
import numpy as np

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


INPUT_PATH = "data/processed/air_quality_temporal.csv"
MODEL_PATH = "artifacts/temporal_air_quality_model.pkl"


FEATURES = [
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
    "pm2_5_lag_1",
    "pm2_5_lag_3",
    "pm2_5_lag_6",
    "pm2_5_lag_24",
    "pm10_lag_1",
    "pm10_lag_3",
    "pm10_lag_6",
    "pm10_lag_24",
    "pm2_5_rolling_6",
    "pm2_5_rolling_24",
    "pm10_rolling_6",
    "pm10_rolling_24",
    "no2_lag_1",
    "co_lag_1",
    "o3_lag_1",
]

TARGETS = [
    "pm2_5_future",
    "pm10_future"
]


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
        "model": model
    }


def main():

    print("=" * 60)
    print("AirTwinNet Temporal Future Forecasting")
    print("=" * 60)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Temporal dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"\nDataset shape: {df.shape}")

    X = df[FEATURES]
    y_pm25 = df["pm2_5_future"]
    y_pm10 = df["pm10_future"]

    # Chronological split
    split_index = int(len(df) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_pm25_train = y_pm25.iloc[:split_index]
    y_pm25_test = y_pm25.iloc[split_index:]

    y_pm10_train = y_pm10.iloc[:split_index]
    y_pm10_test = y_pm10.iloc[split_index:]

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror"
        )

    results = {
        "pm25": {},
        "pm10": {}
    }

    best_pm25 = None
    best_pm10 = None

    print("\n" + "=" * 60)
    print("PM2.5 FUTURE FORECASTING")
    print("=" * 60)

    for name, model in models.items():

        print(f"\nTraining {name}...")

        result = evaluate_model(
            model,
            X_train,
            X_test,
            y_pm25_train,
            y_pm25_test
        )

        results["pm25"][name] = {
            "MAE": result["MAE"],
            "RMSE": result["RMSE"],
            "R2": result["R2"]
        }

        print(
            f"MAE={result['MAE']:.4f} | "
            f"RMSE={result['RMSE']:.4f} | "
            f"R²={result['R2']:.4f}"
        )

        if best_pm25 is None or result["R2"] > best_pm25["R2"]:
            best_pm25 = {
                "name": name,
                "model": result["model"],
                "R2": result["R2"]
            }

    print("\n" + "=" * 60)
    print("PM10 FUTURE FORECASTING")
    print("=" * 60)

    for name, model in models.items():

        print(f"\nTraining {name}...")

        # Fresh model for PM10
        if name == "Random Forest":
            model_pm10 = RandomForestRegressor(
                n_estimators=150,
                random_state=42,
                n_jobs=-1
            )
        elif name == "Extra Trees":
            model_pm10 = ExtraTreesRegressor(
                n_estimators=150,
                random_state=42,
                n_jobs=-1
            )
        elif name == "Gradient Boosting":
            model_pm10 = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
        else:
            model_pm10 = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                objective="reg:squarederror"
            )

        result = evaluate_model(
            model_pm10,
            X_train,
            X_test,
            y_pm10_train,
            y_pm10_test
        )

        results["pm10"][name] = {
            "MAE": result["MAE"],
            "RMSE": result["RMSE"],
            "R2": result["R2"]
        }

        print(
            f"MAE={result['MAE']:.4f} | "
            f"RMSE={result['RMSE']:.4f} | "
            f"R²={result['R2']:.4f}"
        )

        if best_pm10 is None or result["R2"] > best_pm10["R2"]:
            best_pm10 = {
                "name": name,
                "model": result["model"],
                "R2": result["R2"]
            }

    os.makedirs("artifacts", exist_ok=True)

    model_package = {
        "pm25_model": best_pm25["model"],
        "pm10_model": best_pm10["model"],
        "features": FEATURES,
        "pm25_model_name": best_pm25["name"],
        "pm10_model_name": best_pm10["name"],
        "results": results
    }

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model_package, file)

    print("\n" + "=" * 60)
    print("BEST MODELS")
    print("=" * 60)

    print(f"PM2.5 model: {best_pm25['name']}")
    print(f"PM10 model : {best_pm10['name']}")

    print("\nModel saved successfully:")
    print(MODEL_PATH)

    print("\n" + "=" * 60)
    print("TEMPORAL FORECASTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()