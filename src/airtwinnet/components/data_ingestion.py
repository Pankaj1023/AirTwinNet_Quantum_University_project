import os
import pandas as pd


class DataIngestion:

    def __init__(self, data_path: str):

        self.data_path = data_path

    def initiate_data_ingestion(self):

        print("Starting data ingestion...")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset not found at: {self.data_path}"
            )

        data = pd.read_csv(self.data_path)

        print("Data ingestion completed successfully.")
        print(f"Dataset shape: {data.shape}")

        return data