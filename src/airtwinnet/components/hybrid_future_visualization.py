import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


INPUT_PATH = (
    "artifacts/hybrid_future_test_predictions.csv"
)

RESULTS_PATH = (
    "artifacts/hybrid_future_evaluation_results.csv"
)

OUTPUT_DIR = (
    "artifacts/hybrid_future_plots"
)


def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, rmse, r2


def actual_vs_predicted_plot(
    actual,
    predicted,
    title,
    ylabel,
    filename
):

    plt.figure(figsize=(10, 6))

    plt.scatter(
        actual,
        predicted,
        alpha=0.25,
        s=10
    )

    minimum = min(
        actual.min(),
        predicted.min()
    )

    maximum = max(
        actual.max(),
        predicted.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel(
        f"Actual {ylabel}"
    )

    plt.ylabel(
        f"Predicted {ylabel}"
    )

    plt.title(title)

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=150
    )

    plt.close()


def model_comparison_plot():

    if not os.path.exists(
        RESULTS_PATH
    ):
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_PATH}"
        )

    results = pd.read_csv(
        RESULTS_PATH
    )

    models = results["Model"].tolist()

    pm25_r2 = results[
        "PM2.5_R2"
    ].tolist()

    pm10_r2 = results[
        "PM10_R2"
    ].tolist()

    x = np.arange(
        len(models)
    )

    width = 0.35

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width / 2,
        pm25_r2,
        width,
        label="PM2.5 R²"
    )

    plt.bar(
        x + width / 2,
        pm10_r2,
        width,
        label="PM10 R²"
    )

    plt.xticks(
        x,
        models
    )

    plt.ylabel(
        "R² Score"
    )

    plt.xlabel(
        "Model"
    )

    plt.title(
        "Future Forecast Model Comparison"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "model_comparison_r2.png"
        ),
        dpi=150
    )

    plt.close()


def city_wise_plot():

    city_path = (
        "artifacts/"
        "hybrid_future_city_wise_results.csv"
    )

    if not os.path.exists(
        city_path
    ):
        raise FileNotFoundError(
            f"City results not found: {city_path}"
        )

    data = pd.read_csv(
        city_path
    )

    cities = data[
        "city"
    ].tolist()

    pm25_r2 = data[
        "pm25_r2"
    ].tolist()

    pm10_r2 = data[
        "pm10_r2"
    ].tolist()

    x = np.arange(
        len(cities)
    )

    width = 0.35

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width / 2,
        pm25_r2,
        width,
        label="PM2.5 R²"
    )

    plt.bar(
        x + width / 2,
        pm10_r2,
        width,
        label="PM10 R²"
    )

    plt.xticks(
        x,
        cities,
        rotation=20
    )

    plt.ylabel(
        "R² Score"
    )

    plt.xlabel(
        "City"
    )

    plt.title(
        "City-wise Future Hybrid Performance"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "city_wise_hybrid_r2.png"
        ),
        dpi=150
    )

    plt.close()


def time_series_plot(
    data,
    actual_column,
    predicted_column,
    title,
    ylabel,
    filename
):

    data = data.sort_values(
        "timestamp"
    )

    data = data.iloc[
        :1000
    ]

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        data["timestamp"],
        data[actual_column],
        label="Actual"
    )

    plt.plot(
        data["timestamp"],
        data[predicted_column],
        label="Hybrid Predicted"
    )

    plt.xlabel(
        "Timestamp"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.xticks(
        rotation=30
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=150
    )

    plt.close()


def main():

    print("=" * 60)
    print(
        "AirTwinNet Hybrid Future Visualization"
    )
    print("=" * 60)

    if not os.path.exists(
        INPUT_PATH
    ):
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_PATH}"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    data = pd.read_csv(
        INPUT_PATH
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    print()
    print("Prediction data loaded")
    print("-" * 60)
    print(
        f"Rows: {len(data)}"
    )
    print(
        f"Cities: {data['city'].nunique()}"
    )

    # --------------------------------------------------
    # PM2.5 ACTUAL VS PREDICTED
    # --------------------------------------------------

    actual_vs_predicted_plot(
        data["actual_pm2_5"],
        data["hybrid_pm2_5"],
        "Hybrid Future Forecast: PM2.5",
        "PM2.5",
        "hybrid_pm25_actual_vs_predicted.png"
    )

    # --------------------------------------------------
    # PM10 ACTUAL VS PREDICTED
    # --------------------------------------------------

    actual_vs_predicted_plot(
        data["actual_pm10"],
        data["hybrid_pm10"],
        "Hybrid Future Forecast: PM10",
        "PM10",
        "hybrid_pm10_actual_vs_predicted.png"
    )

    # --------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------

    model_comparison_plot()

    # --------------------------------------------------
    # CITY-WISE
    # --------------------------------------------------

    city_wise_plot()

    # --------------------------------------------------
    # TIME SERIES PM2.5
    # --------------------------------------------------

    time_series_plot(
        data,
        "actual_pm2_5",
        "hybrid_pm2_5",
        "PM2.5 Actual vs Hybrid Future Prediction",
        "PM2.5",
        "hybrid_pm25_time_series.png"
    )

    # --------------------------------------------------
    # TIME SERIES PM10
    # --------------------------------------------------

    time_series_plot(
        data,
        "actual_pm10",
        "hybrid_pm10",
        "PM10 Actual vs Hybrid Future Prediction",
        "PM10",
        "hybrid_pm10_time_series.png"
    )

    # --------------------------------------------------
    # PRINT METRICS
    # --------------------------------------------------

    pm25_mae, pm25_rmse, pm25_r2 = (
        calculate_metrics(
            data["actual_pm2_5"],
            data["hybrid_pm2_5"]
        )
    )

    pm10_mae, pm10_rmse, pm10_r2 = (
        calculate_metrics(
            data["actual_pm10"],
            data["hybrid_pm10"]
        )
    )

    print()
    print("=" * 60)
    print("Hybrid Future Forecast Metrics")
    print("=" * 60)

    print(
        f"PM2.5 MAE  : {pm25_mae:.4f}"
    )

    print(
        f"PM2.5 RMSE : {pm25_rmse:.4f}"
    )

    print(
        f"PM2.5 R²   : {pm25_r2:.4f}"
    )

    print()

    print(
        f"PM10 MAE   : {pm10_mae:.4f}"
    )

    print(
        f"PM10 RMSE  : {pm10_rmse:.4f}"
    )

    print(
        f"PM10 R²    : {pm10_r2:.4f}"
    )

    print()
    print("=" * 60)
    print("Plots saved:")
    print("=" * 60)

    for filename in sorted(
        os.listdir(OUTPUT_DIR)
    ):
        print(
            os.path.join(
                OUTPUT_DIR,
                filename
            )
        )

    print()
    print(
        "Hybrid future visualization completed successfully."
    )


if __name__ == "__main__":
    main()