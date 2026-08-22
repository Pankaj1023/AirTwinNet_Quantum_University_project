import pandas as pd


class SpatioTemporalAlignment:

    def __init__(self, data: pd.DataFrame):

        self.data = data.copy()

    def initiate_alignment(self):

        print("Starting spatio-temporal alignment...")

        # Temporal alignment
        if "timestamp" not in self.data.columns:
            raise ValueError(
                "Timestamp column is required for temporal alignment."
            )

        self.data["timestamp"] = pd.to_datetime(
            self.data["timestamp"],
            errors="coerce"
        )

        self.data = self.data.dropna(subset=["timestamp"])

        # Round timestamps to hourly resolution
        self.data["timestamp"] = self.data["timestamp"].dt.floor("h")

        # Spatial alignment
        if "city" not in self.data.columns:
            raise ValueError(
                "City column is required for spatial alignment."
            )

        self.data["city"] = self.data["city"].astype(str).str.strip()

        print("Spatio-temporal alignment completed successfully.")
        print(f"Aligned dataset shape: {self.data.shape}")

        return self.data