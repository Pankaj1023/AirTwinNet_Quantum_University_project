import torch
import torch.nn as nn


class STGNN(nn.Module):

    def __init__(
        self,
        input_features: int,
        hidden_features: int = 32,
        output_features: int = 2
    ):
        super().__init__()

        if input_features <= 0:
            raise ValueError(
                "input_features must be greater than 0."
            )

        if hidden_features <= 0:
            raise ValueError(
                "hidden_features must be greater than 0."
            )

        if output_features <= 0:
            raise ValueError(
                "output_features must be greater than 0."
            )

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features

        # Spatial feature transformation
        self.spatial_layer = nn.Linear(
            input_features,
            hidden_features
        )

        # Temporal feature transformation
        self.temporal_layer = nn.GRU(
            input_size=hidden_features,
            hidden_size=hidden_features,
            batch_first=True
        )

        # Final prediction layer
        self.output_layer = nn.Linear(
            hidden_features,
            output_features
        )

        self.activation = nn.ReLU()

    def spatial_learning(self, node_features):

        if node_features.ndim != 3:
            raise ValueError(
                "Node features must have shape "
                "(batch, time, features)."
            )

        spatial_representation = self.spatial_layer(
            node_features
        )

        spatial_representation = self.activation(
            spatial_representation
        )

        return spatial_representation

    def temporal_learning(self, spatial_representation):

        if spatial_representation.ndim != 3:
            raise ValueError(
                "Spatial representation must have shape "
                "(batch, time, features)."
            )

        temporal_representation, _ = self.temporal_layer(
            spatial_representation
        )

        return temporal_representation

    def forward(self, node_features):

        spatial_representation = self.spatial_learning(
            node_features
        )

        temporal_representation = self.temporal_learning(
            spatial_representation
        )

        # Use the final time step
        final_representation = temporal_representation[:, -1, :]

        prediction = self.output_layer(
            final_representation
        )

        return prediction

    def generate_representation(self, node_features):

        spatial_representation = self.spatial_learning(
            node_features
        )

        temporal_representation = self.temporal_learning(
            spatial_representation
        )

        return temporal_representation