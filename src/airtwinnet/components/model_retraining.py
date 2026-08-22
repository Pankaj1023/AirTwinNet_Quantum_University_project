import numpy as np


class ModelRetraining:

    def __init__(self, error_threshold: float = 2.0):

        if error_threshold < 0:
            raise ValueError(
                "Error threshold cannot be negative."
            )

        self.error_threshold = error_threshold

    def should_retrain(self, mae: float) -> bool:

        if mae < 0:
            raise ValueError(
                "MAE cannot be negative."
            )

        return mae > self.error_threshold

    def prepare_retraining_data(
        self,
        features,
        targets
    ):

        features = np.asarray(
            features,
            dtype=float
        )

        targets = np.asarray(
            targets,
            dtype=float
        )

        if len(features) != len(targets):
            raise ValueError(
                "Features and targets must contain "
                "the same number of samples."
            )

        return {
            "features": features,
            "targets": targets
        }

    def update_model(
        self,
        model,
        training_data
    ):

        if model is None:
            raise ValueError(
                "Model cannot be None."
            )

        if not training_data:
            raise ValueError(
                "Training data cannot be empty."
            )

        # Foundation hook for actual model retraining.
        model.retraining_status = "updated"

        return model