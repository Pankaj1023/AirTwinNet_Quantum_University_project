import numpy as np
import torch

from src.airtwinnet.components.st_gnn import STGNN


def test_st_gnn():

    node_features = np.array([
        [
            [30.0, 70.0, 42.3, 68.1, 24.5, 0.71, 31.2],
            [31.0, 66.0, 35.2, 57.8, 20.4, 0.64, 38.1],
        ]
    ], dtype=np.float32)

    node_features = torch.tensor(node_features)

    model = STGNN(
        input_features=7,
        hidden_features=16
    )

    representation = model.generate_representation(
        node_features
    )

    assert representation.shape == (1, 2, 16)

    output = model(node_features)

    assert output.shape == (1, 2)