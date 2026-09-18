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
OUTPUT_PATH = "artifacts/temporal_city_wise_performance.csv"


def main():

    print("=" * 60)
    print("AirTwinNet City-wise Temporal Evaluation")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Load trained models
    with open(MODEL_PATH, "rb") as file:
        package = pickle.load(file)

    features = package["features"]
    pm25_model = package["pm25_model"]
    pm10_model = package["pm10_model"]

    # Same chronological 80/20 split
    split_index = int(len(df) * 0.80)

    test_df = df.iloc[split_index:].copy()

    print(f"\nTesting rows: {len(test_df)}")

    # Predictions
    X_test = test_df[features]

    test_df["pm2_5_predicted"] = pm25_model.predict(X_test)
    test_df["pm10_predicted"] = pm10_model.predict(X_test)

    cities = sorted(test_df["city"].unique())

    results = []

    for city in cities:

        city_df = test_df[test_df["city"] == city]

        actual_pm25 = city_df["pm2_5_future"]
        predicted_pm25 = city_df["pm2_5_predicted"]

        actual_pm10 = city_df["pm10_future"]
        predicted_pm10 = city_df["pm10_predicted"]

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

            "pm25_mae": round(pm25_mae, 4),
            "pm25_rmse": round(pm25_rmse, 4),
            "pm25_r2": round(pm25_r2, 4),

            "pm10_mae": round(pm10_mae, 4),
            "pm10_rmse": round(pm10_rmse, 4),
            "pm10_r2": round(pm10_r2, 4),

            "test_samples": len(city_df)
        })

    results_df = pd.DataFrame(results)

    # Save results
    os.makedirs("artifacts", exist_ok=True)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # Display
    print("\nCity-wise Performance:")
    print(results_df.to_string(index=False))

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("CITY-WISE TEMPORAL EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()