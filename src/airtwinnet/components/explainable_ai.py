import numpy as np


class ExplainableAI:

    def __init__(self, feature_names):

        if not feature_names:
            raise ValueError(
                "Feature names cannot be empty."
            )

        self.feature_names = feature_names

    def calculate_feature_contribution(
        self,
        feature_values,
        feature_weights
    ):

        feature_values = np.asarray(
            feature_values,
            dtype=float
        )

        feature_weights = np.asarray(
            feature_weights,
            dtype=float
        )

        if feature_values.ndim != 2:
            raise ValueError(
                "Feature values must be a 2D array."
            )

        if feature_weights.ndim != 1:
            raise ValueError(
                "Feature weights must be a 1D array."
            )

        if feature_values.shape[1] != len(
            self.feature_names
        ):
            raise ValueError(
                "Feature count does not match feature names."
            )

        if len(feature_weights) != len(
            self.feature_names
        ):
            raise ValueError(
                "Weight count does not match feature names."
            )

        contributions = (
            feature_values * feature_weights
        )

        return contributions

    def get_feature_importance(
        self,
        feature_values,
        feature_weights
    ):

        contributions = self.calculate_feature_contribution(
            feature_values,
            feature_weights
        )

        importance = np.mean(
            np.abs(contributions),
            axis=0
        )

        total = importance.sum()

        if total > 0:
            importance = importance / total

        return dict(
            zip(
                self.feature_names,
                importance
            )
        )