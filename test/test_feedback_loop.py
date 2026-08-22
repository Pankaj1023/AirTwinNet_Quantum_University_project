import numpy as np

from src.airtwinnet.components.error_monitoring import (
    ErrorMonitoring
)

from src.airtwinnet.components.model_retraining import (
    ModelRetraining
)

from src.airtwinnet.components.feedback_loop import (
    AirTwinNetFeedbackLoop
)


class DummyModel:
    pass


def test_feedback_loop():

    error_monitor = ErrorMonitoring()

    retraining = ModelRetraining(
        error_threshold=2.0
    )

    feedback_loop = AirTwinNetFeedbackLoop(
        error_monitor,
        retraining
    )

    actual = np.array([
        50.0,
        60.0,
        70.0
    ])

    predicted = np.array([
        40.0,
        45.0,
        55.0
    ])

    performance = feedback_loop.process_feedback(
        actual,
        predicted
    )

    assert "metrics" in performance

    assert "retraining_required" in performance

    model = DummyModel()

    training_data = {
        "features": np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ]),
        "targets": np.array([
            10.0,
            20.0
        ])
    }

    result = feedback_loop.execute_retraining_if_required(
        model,
        training_data,
        performance
    )

    assert "retrained" in result

    assert result["retrained"] is True

    print("Feedback loop test passed successfully.")