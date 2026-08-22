import pandas as pd


class DynamicUrbanAirQualityGraph:

    def __init__(self, digital_twin_state: pd.DataFrame):

        self.data = digital_twin_state.copy()

    def build_graph(self):

        print("Starting dynamic graph construction...")

        if self.data.empty:
            raise ValueError(
                "Digital Twin state cannot be empty."
            )

        required_columns = [
            "timestamp",
            "city"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Required graph columns missing: {missing_columns}"
            )

        # Create unique graph nodes from spatial locations
        nodes = (
            self.data["city"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        # Create simple spatial relationships
        edges = []

        for index in range(len(nodes) - 1):

            edges.append(
                (nodes[index], nodes[index + 1])
            )

        # Node features
        feature_columns = [
            column
            for column in self.data.columns
            if column not in ["timestamp", "city"]
        ]

        node_features = {}

        for node in nodes:

            node_data = self.data[
                self.data["city"] == node
            ]

            node_features[node] = (
                node_data[feature_columns]
                .mean()
                .to_dict()
            )

        graph = {
            "nodes": nodes,
            "edges": edges,
            "node_features": node_features
        }

        print("Dynamic graph construction completed successfully.")
        print(f"Number of nodes: {len(nodes)}")
        print(f"Number of edges: {len(edges)}")

        return graph