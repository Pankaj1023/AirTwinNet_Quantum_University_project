import pandas as pd

from src.airtwinnet.components.data_preprocessing import DataPreprocessing


def test_data_preprocessing():

    data_path = "data/raw/air_quality_sample.csv"

    data = pd.read_csv(data_path)

    preprocessing = DataPreprocessing(data)

    processed_data = preprocessing.initiate_data_preprocessing()

    assert processed_data is not None
    assert not processed_data.empty

    assert "timestamp" in processed_data.columns
    assert pd.api.types.is_datetime64_any_dtype(
        processed_data["timestamp"]
    )

    print("Data preprocessing test passed successfully.")
    print(f"Processed rows: {processed_data.shape[0]}")
    print(f"Processed columns: {processed_data.shape[1]}")