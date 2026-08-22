import numpy as np

from src.airtwinnet.components.model_retraining import (
    ModelRetraining
)


class DummyModel:
    pass


def test_model_retraining():

    retraining = ModelRetraining(
        error_threshold=2.0
    )

    assert retraining.should_retrain(3.0) is True

    assert retraining.should_retrain(1.0) is False

    features = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    targets = np.array([
        10.0,
        20.0
    ])

    training_data = retraining.prepare_retraining_data(
        features,
        targets
    )

    assert "features" in training_data
    assert "targets" in training_data

    model = DummyModel()

    updated_model = retraining.update_model(
        model,
        training_data
    )

    assert updated_model.retraining_status == "updated"

    print("Model retraining test passed successfully.")