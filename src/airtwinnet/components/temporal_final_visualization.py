import os
import pandas as pd
import matplotlib.pyplot as plt


INPUT_PATH = "artifacts/temporal_city_wise_performance_v2.csv"
OUTPUT_DIR = "artifacts/temporal_final_plots"


def create_bar_plot(df, columns, title, ylabel, filename):

    plt.figure(figsize=(10, 6))

    x = range(len(df))

    width = 0.35

    for i, column in enumerate(columns):
        values = df[column].values

        positions = [
            value + (i - 0.5) * width
            for value in x
        ]

        plt.bar(
            positions,
            values,
            width=width,
            label=column.upper()
        )

    plt.xticks(
        list(x),
        df["city"]
    )

    plt.title(title)
    plt.xlabel("City")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=150
    )

    plt.close()


def main():

    print("=" * 60)
    print("AirTwinNet Final Temporal Visualization")
    print("=" * 60)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df = pd.read_csv(INPUT_PATH)

    # --------------------------------------------------
    # 1. R2 Comparison
    # --------------------------------------------------

    create_bar_plot(
        df,
        ["pm25_r2", "pm10_r2"],
        "City-wise Temporal Forecasting R² Comparison",
        "R² Score",
        "city_wise_r2_comparison.png"
    )

    # --------------------------------------------------
    # 2. MAE Comparison
    # --------------------------------------------------

    create_bar_plot(
        df,
        ["pm25_mae", "pm10_mae"],
        "City-wise Temporal Forecasting MAE Comparison",
        "MAE",
        "city_wise_mae_comparison.png"
    )

    # --------------------------------------------------
    # 3. RMSE Comparison
    # --------------------------------------------------

    create_bar_plot(
        df,
        ["pm25_rmse", "pm10_rmse"],
        "City-wise Temporal Forecasting RMSE Comparison",
        "RMSE",
        "city_wise_rmse_comparison.png"
    )

    # --------------------------------------------------
    # Overall averages
    # --------------------------------------------------

    overall = {
        "pm25_mae": df["pm25_mae"].mean(),
        "pm25_rmse": df["pm25_rmse"].mean(),
        "pm25_r2": df["pm25_r2"].mean(),

        "pm10_mae": df["pm10_mae"].mean(),
        "pm10_rmse": df["pm10_rmse"].mean(),
        "pm10_r2": df["pm10_r2"].mean()
    }

    overall_df = pd.DataFrame(
        [overall]
    )

    overall_path = (
        "artifacts/"
        "temporal_overall_city_average.csv"
    )

    overall_df.to_csv(
        overall_path,
        index=False
    )

    print("\nOverall City Average:")
    print(
        overall_df.to_string(index=False)
    )

    print("\nPlots saved in:")
    print(OUTPUT_DIR)

    print("\nOverall results saved:")
    print(overall_path)

    print("\n" + "=" * 60)
    print("FINAL TEMPORAL VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()