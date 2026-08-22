import numpy as np


class MultiHybridPredictionEngine:

    def __init__(
        self,
        st_gnn_features: int,
        output_features: int = 3
    ):

        self.st_gnn_features = st_gnn_features
        self.output_features = output_features

        self.weights = np.random.randn(
            st_gnn_features,
            output_features
        ) * 0.01

        self.bias = np.zeros(output_features)

    def predict(self, st_gnn_representation):

        representation = np.asarray(
            st_gnn_representation,
            dtype=float
        )

        if representation.ndim != 2:
            raise ValueError(
                "ST-GNN representation must be a 2D array."
            )

        predictions = (
            representation @ self.weights
            + self.bias
        )

        return predictions

    def predict_with_uncertainty(
        self,
        st_gnn_representation
    ):

        predictions = self.predict(
            st_gnn_representation
        )

        # Initial uncertainty estimate based on
        # prediction dispersion.
        uncertainty = np.std(
            predictions,
            axis=1,
            keepdims=True
        )

        confidence = 1.0 / (
            1.0 + uncertainty
        )

        return {
            "predictions": predictions,
            "uncertainty": uncertainty,
            "confidence": confidence
        }