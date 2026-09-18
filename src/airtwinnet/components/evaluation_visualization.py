import os
import pandas as pd
import matplotlib.pyplot as plt


class EvaluationVisualizer:
    """
    AirTwinNet model evaluation visualization module.

    Generates:
    1. PM2.5 Actual vs Predicted
    2. PM10 Actual vs Predicted
    3. PM2.5 Residual Analysis
    4. PM10 Residual Analysis
    5. City-wise R2 Comparison
    6. Model Performance Comparison
    """

    def __init__(self):
        self.artifacts_dir = "artifacts"

        self.output_dir = os.path.join(
            self.artifacts_dir,
            "evaluation_plots"
        )

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        self.predictions_file = os.path.join(
            self.artifacts_dir,
            "model_predictions.csv"
        )

        self.city_file = os.path.join(
            self.artifacts_dir,
            "city_wise_evaluation.csv"
        )

    def load_data(self):
        """Load evaluation artifacts."""

        if not os.path.exists(self.predictions_file):
            raise FileNotFoundError(
                f"Prediction file not found: {self.predictions_file}"
            )

        if not os.path.exists(self.city_file):
            raise FileNotFoundError(
                f"City evaluation file not found: {self.city_file}"
            )

        predictions = pd.read_csv(
            self.predictions_file
        )

        city_results = pd.read_csv(
            self.city_file
        )

        return predictions, city_results

    def plot_actual_vs_predicted(
        self,
        data,
        actual_column,
        predicted_column,
        title,
        filename
    ):
        """Create Actual vs Predicted plot."""

        plot_data = data[
            [actual_column, predicted_column]
        ].dropna()

        # Downsample for clean visualization
        if len(plot_data) > 2000:
            plot_data = plot_data.sample(
                2000,
                random_state=42
            ).sort_index()

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            plot_data[actual_column].values,
            label="Actual"
        )

        plt.plot(
            plot_data[predicted_column].values,
            label="Predicted"
        )

        plt.title(title)
        plt.xlabel("Test Samples")
        plt.ylabel("Concentration")

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        output_path = os.path.join(
            self.output_dir,
            filename
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )

    def plot_residual_analysis(
        self,
        data,
        actual_column,
        predicted_column,
        title,
        filename
    ):
        """Create residual/error analysis."""

        plot_data = data[
            [actual_column, predicted_column]
        ].dropna()

        if len(plot_data) > 2000:
            plot_data = plot_data.sample(
                2000,
                random_state=42
            )

        residuals = (
            plot_data[actual_column]
            - plot_data[predicted_column]
        )

        plt.figure(
            figsize=(10, 6)
        )

        plt.scatter(
            plot_data[actual_column],
            residuals,
            alpha=0.5
        )

        plt.axhline(
            y=0,
            linestyle="--"
        )

        plt.title(title)

        plt.xlabel(
            "Actual Value"
        )

        plt.ylabel(
            "Residual (Actual - Predicted)"
        )

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        output_path = os.path.join(
            self.output_dir,
            filename
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )

    def plot_city_wise_r2(
        self,
        city_results
    ):
        """Create city-wise R2 comparison."""

        required_columns = {
            "city",
            "pm2_5_r2",
            "pm10_r2"
        }

        if not required_columns.issubset(
            city_results.columns
        ):
            raise ValueError(
                "City evaluation CSV does not contain "
                "required R2 columns."
            )

        plt.figure(
            figsize=(11, 6)
        )

        x = range(
            len(city_results)
        )

        width = 0.35

        plt.bar(
            [
                i - width / 2
                for i in x
            ],
            city_results["pm2_5_r2"],
            width=width,
            label="PM2.5 R²"
        )

        plt.bar(
            [
                i + width / 2
                for i in x
            ],
            city_results["pm10_r2"],
            width=width,
            label="PM10 R²"
        )

        plt.xticks(
            list(x),
            city_results["city"],
            rotation=30
        )

        plt.ylabel(
            "R² Score"
        )

        plt.xlabel(
            "City"
        )

        plt.title(
            "City-wise Model Performance (R²)"
        )

        plt.legend()

        plt.grid(
            axis="y",
            alpha=0.3
        )

        plt.tight_layout()

        output_path = os.path.join(
            self.output_dir,
            "city_wise_r2_comparison.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )

    def plot_model_comparison(self):
        """
        Visualize model comparison results
        from the latest model improvement experiment.
        """

        models = [
            "RandomForest",
            "ExtraTrees",
            "GradientBoosting",
            "XGBoost"
        ]

        pm25_r2 = [
            0.7389,
            0.7388,
            0.7501,
            0.7500
        ]

        pm10_r2 = [
            0.3620,
            0.3582,
            0.3851,
            0.3825
        ]

        x = range(
            len(models)
        )

        width = 0.35

        plt.figure(
            figsize=(11, 6)
        )

        plt.bar(
            [
                i - width / 2
                for i in x
            ],
            pm25_r2,
            width=width,
            label="PM2.5 R²"
        )

        plt.bar(
            [
                i + width / 2
                for i in x
            ],
            pm10_r2,
            width=width,
            label="PM10 R²"
        )

        plt.xticks(
            list(x),
            models,
            rotation=20
        )

        plt.ylabel(
            "R² Score"
        )

        plt.xlabel(
            "Model"
        )

        plt.title(
            "Model Performance Comparison"
        )

        plt.legend()

        plt.grid(
            axis="y",
            alpha=0.3
        )

        plt.tight_layout()

        output_path = os.path.join(
            self.output_dir,
            "model_performance_comparison.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )

    def generate_all_plots(self):
        """Generate all evaluation visualizations."""

        print(
            "\n" + "=" * 60
        )

        print(
            "AirTwinNet Evaluation Visualization"
        )

        print(
            "=" * 60
        )

        predictions, city_results = self.load_data()

        print(
            f"\nPrediction records loaded: "
            f"{len(predictions)}"
        )

        print(
            f"City records loaded: "
            f"{len(city_results)}"
        )

        # --------------------------------------------------
        # 1. PM2.5 Actual vs Predicted
        # --------------------------------------------------

        self.plot_actual_vs_predicted(
            predictions,
            "pm2_5",
            "predicted_pm2_5",
            "PM2.5 - Actual vs Predicted",
            "pm25_actual_vs_predicted.png"
        )

        # --------------------------------------------------
        # 2. PM10 Actual vs Predicted
        # --------------------------------------------------

        self.plot_actual_vs_predicted(
            predictions,
            "pm10",
            "predicted_pm10",
            "PM10 - Actual vs Predicted",
            "pm10_actual_vs_predicted.png"
        )

        # --------------------------------------------------
        # 3. PM2.5 Residual Analysis
        # --------------------------------------------------

        self.plot_residual_analysis(
            predictions,
            "pm2_5",
            "predicted_pm2_5",
            "PM2.5 Residual Analysis",
            "pm25_residual_analysis.png"
        )

        # --------------------------------------------------
        # 4. PM10 Residual Analysis
        # --------------------------------------------------

        self.plot_residual_analysis(
            predictions,
            "pm10",
            "predicted_pm10",
            "PM10 Residual Analysis",
            "pm10_residual_analysis.png"
        )

        # --------------------------------------------------
        # 5. City-wise R2 Comparison
        # --------------------------------------------------

        self.plot_city_wise_r2(
            city_results
        )

        # --------------------------------------------------
        # 6. Model Performance Comparison
        # --------------------------------------------------

        self.plot_model_comparison()

        print(
            "\n" + "=" * 60
        )

        print(
            "ALL EVALUATION PLOTS GENERATED SUCCESSFULLY"
        )

        print(
            "=" * 60
        )

        print(
            f"\nOutput directory:\n"
            f"{self.output_dir}"
        )


if __name__ == "__main__":

    visualizer = EvaluationVisualizer()

    visualizer.generate_all_plots()