import numpy as np

from src.airtwinnet.components.st_gnn import STGNN


def test_st_gnn():

    node_features = np.array([
        [30.0, 70.0, 42.3, 68.1, 24.5, 0.71, 31.2],
        [31.0, 66.0, 35.2, 57.8, 20.4, 0.64, 38.1],
    ])

    model = STGNN(
        input_features=7,
        hidden_features=16
    )

    representation = model.generate_representation(
        node_features
    )

    assert representation is not None

    assert representation.shape == (2, 16)

    assert np.isfinite(representation).all()

    print("ST-GNN test passed successfully.")
    print(
        f"ST-GNN representation shape: "
        f"{representation.shape}"
    )