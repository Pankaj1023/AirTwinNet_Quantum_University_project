import numpy as np

from src.airtwinnet.components.explainable_ai import (
    ExplainableAI
)


def test_explainable_ai():

    feature_names = [
        "temperature",
        "humidity",
        "pm2_5",
        "pm10",
        "no2",
        "co",
        "o3"
    ]

    feature_values = np.array([
        [30.0, 70.0, 42.3, 68.1, 24.5, 0.71, 31.2],
        [31.0, 66.0, 35.2, 57.8, 20.4, 0.64, 38.1]
    ])

    feature_weights = np.array([
        0.10,
        0.15,
        0.25,
        0.20,
        0.15,
        0.05,
        0.10
    ])

    xai = ExplainableAI(feature_names)

    contributions = xai.calculate_feature_contribution(
        feature_values,
        feature_weights
    )

    importance = xai.get_feature_importance(
        feature_values,
        feature_weights
    )

    assert contributions.shape == feature_values.shape

    assert importance is not None

    assert len(importance) == 7

    assert set(importance.keys()) == set(
        feature_names
    )

    assert np.isfinite(
        list(importance.values())
    ).all()

    print("Explainable AI test passed successfully.")
    print("Feature importance:", importance)