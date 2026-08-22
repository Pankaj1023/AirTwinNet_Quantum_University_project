import numpy as np

from src.airtwinnet.components.error_monitoring import (
    ErrorMonitoring
)


def test_error_monitoring():

    actual = np.array([
        42.0,
        40.0,
        38.0,
        35.0
    ])

    predicted = np.array([
        40.0,
        41.0,
        37.0,
        36.0
    ])

    monitor = ErrorMonitoring()

    errors = monitor.compare_predictions(
        actual,
        predicted
    )

    metrics = monitor.calculate_metrics(
        actual,
        predicted
    )

    performance = monitor.monitor_performance(
        actual,
        predicted,
        error_threshold=2.0
    )

    assert errors.shape == actual.shape

    assert "mae" in metrics
    assert "rmse" in metrics

    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0

    assert "metrics" in performance
    assert "retraining_required" in performance

    assert isinstance(
        performance["retraining_required"],
        bool
    )

    assert np.isfinite(
        metrics["mae"]
    )

    assert np.isfinite(
        metrics["rmse"]
    )

    print(
        "Error monitoring test passed successfully."
    )
    print("Metrics:", metrics)