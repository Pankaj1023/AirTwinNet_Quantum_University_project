import numpy as np


class STGNN:

    def __init__(self, input_features: int, hidden_features: int = 16):

        self.input_features = input_features
        self.hidden_features = hidden_features

        # Simple learnable parameters for the initial ST-GNN foundation
        self.spatial_weights = np.random.randn(
            input_features,
            hidden_features
        ) * 0.01

        self.temporal_weights = np.random.randn(
            hidden_features,
            hidden_features
        ) * 0.01

    def spatial_learning(self, node_features):

        node_features = np.asarray(node_features, dtype=float)

        if node_features.ndim != 2:
            raise ValueError(
                "Node features must be a 2D array."
            )

        spatial_representation = np.tanh(
            node_features @ self.spatial_weights
        )

        return spatial_representation

    def temporal_learning(self, spatial_representation):

        spatial_representation = np.asarray(
            spatial_representation,
            dtype=float
        )

        if spatial_representation.ndim != 2:
            raise ValueError(
                "Spatial representation must be a 2D array."
            )

        temporal_representation = np.tanh(
            spatial_representation @ self.temporal_weights
        )

        return temporal_representation

    def generate_representation(self, node_features):

        spatial_representation = self.spatial_learning(
            node_features
        )

        temporal_representation = self.temporal_learning(
            spatial_representation
        )

        return temporal_representation