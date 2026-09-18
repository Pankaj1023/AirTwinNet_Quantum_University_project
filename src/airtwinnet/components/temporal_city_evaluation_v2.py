import os
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


DATA_PATH = "data/processed/air_quality_temporal.csv"
MODEL_PATH = "artifacts/temporal_air_quality_model.pkl"
OUTPUT_PATH = "artifacts/temporal_city_wise_performance_v2.csv"


def main():

    print("=" * 60)
    print("AirTwinNet Proper City-wise Temporal Evaluation")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Load trained models
    with open(MODEL_PATH, "rb") as file:
        package = pickle.load(file)

    features = package["features"]
    pm25_model = package["pm25_model"]
    pm10_model = package["pm10_model"]

    results = []

    cities = sorted(df["city"].unique())

    print(f"\nCities found: {cities}")

    for city in cities:

        print(f"\nEvaluating: {city}")

        city_df = df[df["city"] == city].copy()

        # Sort chronologically
        city_df["timestamp"] = pd.to_datetime(
            city_df["timestamp"]
        )

        city_df = city_df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        # 80/20 chronological split
        split_index = int(len(city_df) * 0.80)

        train_df = city_df.iloc[:split_index]
        test_df = city_df.iloc[split_index:]

        X_test = test_df[features]

        actual_pm25 = test_df["pm2_5_future"].values
        actual_pm10 = test_df["pm10_future"].values

        predicted_pm25 = pm25_model.predict(X_test)
        predicted_pm10 = pm10_model.predict(X_test)

        # PM2.5 metrics
        pm25_mae = mean_absolute_error(
            actual_pm25,
            predicted_pm25
        )

        pm25_rmse = np.sqrt(
            mean_squared_error(
                actual_pm25,
                predicted_pm25
            )
        )

        pm25_r2 = r2_score(
            actual_pm25,
            predicted_pm25
        )

        # PM10 metrics
        pm10_mae = mean_absolute_error(
            actual_pm10,
            predicted_pm10
        )

        pm10_rmse = np.sqrt(
            mean_squared_error(
                actual_pm10,
                predicted_pm10
            )
        )

        pm10_r2 = r2_score(
            actual_pm10,
            predicted_pm10
        )

        results.append({
            "city": city,

            "train_samples": len(train_df),
            "test_samples": len(test_df),

            "pm25_mae": round(pm25_mae, 4),
            "pm25_rmse": round(pm25_rmse, 4),
            "pm25_r2": round(pm25_r2, 4),

            "pm10_mae": round(pm10_mae, 4),
            "pm10_rmse": round(pm10_rmse, 4),
            "pm10_r2": round(pm10_r2, 4)
        })

        print(
            f"  Train: {len(train_df)} | "
            f"Test: {len(test_df)}"
        )

        print(
            f"  PM2.5 R²: {pm25_r2:.4f}"
        )

        print(
            f"  PM10 R²: {pm10_r2:.4f}"
        )

    # Create result dataframe
    results_df = pd.DataFrame(results)

    # Save
    os.makedirs("artifacts", exist_ok=True)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("FINAL CITY-WISE RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(index=False)
    )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("PROPER CITY-WISE EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()