import numpy as np


class ErrorMonitoring:

    def compare_predictions(
        self,
        actual,
        predicted
    ):

        actual = np.asarray(
            actual,
            dtype=float
        )

        predicted = np.asarray(
            predicted,
            dtype=float
        )

        if actual.shape != predicted.shape:
            raise ValueError(
                "Actual and predicted values "
                "must have the same shape."
            )

        errors = actual - predicted

        return errors

    def calculate_metrics(
        self,
        actual,
        predicted
    ):

        actual = np.asarray(
            actual,
            dtype=float
        )

        predicted = np.asarray(
            predicted,
            dtype=float
        )

        errors = actual - predicted

        mae = np.mean(
            np.abs(errors)
        )

        rmse = np.sqrt(
            np.mean(errors ** 2)
        )

        return {
            "mae": float(mae),
            "rmse": float(rmse)
        }

    def monitor_performance(
        self,
        actual,
        predicted,
        error_threshold
    ):

        metrics = self.calculate_metrics(
            actual,
            predicted
        )

        retraining_required = (
            metrics["mae"] > error_threshold
        )

        return {
            "metrics": metrics,
            "retraining_required":
                retraining_required
        }