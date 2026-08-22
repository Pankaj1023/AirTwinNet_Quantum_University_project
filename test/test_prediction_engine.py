import numpy as np

from src.airtwinnet.components.prediction_engine import (
    MultiHybridPredictionEngine
)


def test_prediction_engine():

    st_gnn_representation = np.array([
        [0.10] * 16,
        [0.20] * 16
    ])

    engine = MultiHybridPredictionEngine(
        st_gnn_features=16,
        output_features=3
    )

    result = engine.predict_with_uncertainty(
        st_gnn_representation
    )

    assert result is not None

    assert "predictions" in result
    assert "uncertainty" in result
    assert "confidence" in result

    assert result["predictions"].shape == (2, 3)

    assert result["uncertainty"].shape == (2, 1)

    assert result["confidence"].shape == (2, 1)

    assert np.isfinite(
        result["predictions"]
    ).all()

    assert np.isfinite(
        result["confidence"]
    ).all()

    print(
        "Multi-Hybrid Prediction Engine "
        "test passed successfully."
    )

    print(
        f"Prediction shape: "
        f"{result['predictions'].shape}"
    )