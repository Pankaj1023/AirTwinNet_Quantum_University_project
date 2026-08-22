import pandas as pd

from src.airtwinnet.components.graph_construction import (
    DynamicUrbanAirQualityGraph
)


def test_graph_construction():

    data_path = "data/raw/air_quality_sample.csv"

    data = pd.read_csv(data_path)

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    graph_builder = DynamicUrbanAirQualityGraph(data)

    graph = graph_builder.build_graph()

    assert graph is not None

    # Check graph structure
    assert "nodes" in graph
    assert "edges" in graph
    assert "node_features" in graph

    # Current dataset contains Roorkee
    assert "Roorkee" in graph["nodes"]

    # Check node features
    assert "Roorkee" in graph["node_features"]

    expected_features = [
        "temperature",
        "humidity",
        "pm2_5",
        "pm10",
        "no2",
        "co",
        "o3"
    ]

    for feature in expected_features:
        assert feature in graph["node_features"]["Roorkee"]

    print("Dynamic graph construction test passed successfully.")
    print(f"Number of nodes: {len(graph['nodes'])}")
    print(f"Number of edges: {len(graph['edges'])}")