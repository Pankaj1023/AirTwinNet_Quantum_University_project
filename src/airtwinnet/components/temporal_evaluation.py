import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


DATA_PATH = "data/processed/air_quality_temporal.csv"
MODEL_PATH = "artifacts/temporal_air_quality_model.pkl"
OUTPUT_DIR = "artifacts/temporal_evaluation_plots"


def main():

    print("=" * 60)
    print("AirTwinNet Temporal Model Evaluation")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Load trained models
    with open(MODEL_PATH, "rb") as file:
        package = pickle.load(file)

    features = package["features"]
    pm25_model = package["pm25_model"]
    pm10_model = package["pm10_model"]

    split_index = int(len(df) * 0.80)

    test_df = df.iloc[split_index:].copy()

    X_test = test_df[features]

    actual_pm25 = test_df["pm2_5_future"].values
    actual_pm10 = test_df["pm10_future"].values

    predicted_pm25 = pm25_model.predict(X_test)
    predicted_pm10 = pm10_model.predict(X_test)

    # Metrics
    pm25_mae = mean_absolute_error(actual_pm25, predicted_pm25)
    pm25_rmse = np.sqrt(
        mean_squared_error(actual_pm25, predicted_pm25)
    )
    pm25_r2 = r2_score(actual_pm25, predicted_pm25)

    pm10_mae = mean_absolute_error(actual_pm10, predicted_pm10)
    pm10_rmse = np.sqrt(
        mean_squared_error(actual_pm10, predicted_pm10)
    )
    pm10_r2 = r2_score(actual_pm10, predicted_pm10)

    print("\nPM2.5 Future Forecast")
    print(f"MAE  : {pm25_mae:.4f}")
    print(f"RMSE : {pm25_rmse:.4f}")
    print(f"R²   : {pm25_r2:.4f}")

    print("\nPM10 Future Forecast")
    print(f"MAE  : {pm10_mae:.4f}")
    print(f"RMSE : {pm10_rmse:.4f}")
    print(f"R²   : {pm10_r2:.4f}")

    # --------------------------------------------------
    # 1. PM2.5 Actual vs Predicted
    # --------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        actual_pm25[:1000],
        label="Actual PM2.5"
    )

    plt.plot(
        predicted_pm25[:1000],
        label="Predicted PM2.5"
    )

    plt.title("PM2.5 Future Forecast: Actual vs Predicted")
    plt.xlabel("Test Samples")
    plt.ylabel("PM2.5")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "temporal_pm25_actual_vs_predicted.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # 2. PM10 Actual vs Predicted
    # --------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        actual_pm10[:1000],
        label="Actual PM10"
    )

    plt.plot(
        predicted_pm10[:1000],
        label="Predicted PM10"
    )

    plt.title("PM10 Future Forecast: Actual vs Predicted")
    plt.xlabel("Test Samples")
    plt.ylabel("PM10")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "temporal_pm10_actual_vs_predicted.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # 3. PM2.5 Residual Analysis
    # --------------------------------------------------

    pm25_residuals = actual_pm25 - predicted_pm25

    plt.figure(figsize=(10, 6))

    plt.scatter(
        predicted_pm25,
        pm25_residuals,
        alpha=0.3
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.title("PM2.5 Future Forecast Residual Analysis")
    plt.xlabel("Predicted PM2.5")
    plt.ylabel("Residual")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "temporal_pm25_residual_analysis.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # 4. PM10 Residual Analysis
    # --------------------------------------------------

    pm10_residuals = actual_pm10 - predicted_pm10

    plt.figure(figsize=(10, 6))

    plt.scatter(
        predicted_pm10,
        pm10_residuals,
        alpha=0.3
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.title("PM10 Future Forecast Residual Analysis")
    plt.xlabel("Predicted PM10")
    plt.ylabel("Residual")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "temporal_pm10_residual_analysis.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # Save predictions
    # --------------------------------------------------

    results_df = test_df[
        ["timestamp", "city", "pm2_5_future", "pm10_future"]
    ].copy()

    results_df["pm2_5_predicted"] = predicted_pm25
    results_df["pm10_predicted"] = predicted_pm10

    results_path = os.path.join(
        "artifacts",
        "temporal_forecasting_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\nEvaluation plots saved in:")
    print(OUTPUT_DIR)

    print("\nForecast results saved:")
    print(results_path)

    print("\n" + "=" * 60)
    print("TEMPORAL MODEL EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()