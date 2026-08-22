import os

from src.airtwinnet.components.data_ingestion import DataIngestion


def test_data_ingestion():

    data_path = "data/raw/air_quality_sample.csv"

    ingestion = DataIngestion(data_path)

    data = ingestion.initiate_data_ingestion()

    assert data is not None
    assert not data.empty

    print("Data ingestion test passed successfully.")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")