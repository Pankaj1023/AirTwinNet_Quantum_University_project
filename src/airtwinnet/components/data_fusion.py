import pandas as pd


class MultiModalDataFusion:

    def __init__(self, data: pd.DataFrame):

        self.data = data.copy()

    def initiate_data_fusion(self):

        print("Starting multi-modal data fusion...")

        if self.data.empty:
            raise ValueError(
                "Input data cannot be empty."
            )

        # Required spatial and temporal information
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
                f"Required columns missing: {missing_columns}"
            )

        # Ensure timestamp is datetime
        self.data["timestamp"] = pd.to_datetime(
            self.data["timestamp"],
            errors="coerce"
        )

        self.data = self.data.dropna(
            subset=["timestamp"]
        )

        # Ensure spatial identifier is clean
        self.data["city"] = (
            self.data["city"]
            .astype(str)
            .str.strip()
        )

        # Create unified feature representation
        feature_columns = [
            column
            for column in self.data.columns
            if column not in ["timestamp", "city"]
        ]

        if not feature_columns:
            raise ValueError(
                "No feature columns available for data fusion."
            )

        # Keep timestamp + spatial information + all aligned features
        fused_data = self.data[
            ["timestamp", "city"] + feature_columns
        ].copy()

        print("Multi-modal data fusion completed successfully.")
        print(f"Fused dataset shape: {fused_data.shape}")

        return fused_data