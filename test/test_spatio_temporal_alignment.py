import pandas as pd

from src.airtwinnet.components.spatio_temporal_alignment import (
    SpatioTemporalAlignment
)


def test_spatio_temporal_alignment():

    data_path = "data/raw/air_quality_sample.csv"

    data = pd.read_csv(data_path)

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    alignment = SpatioTemporalAlignment(data)

    aligned_data = alignment.initiate_alignment()

    assert aligned_data is not None
    assert not aligned_data.empty

    # Check timestamp alignment
    assert "timestamp" in aligned_data.columns
    assert pd.api.types.is_datetime64_any_dtype(
        aligned_data["timestamp"]
    )

    # Check spatial information
    assert "city" in aligned_data.columns

    # Check that timestamps are aligned to hourly resolution
    assert all(
        aligned_data["timestamp"].dt.minute == 0
    )

    print("Spatio-temporal alignment test passed successfully.")
    print(f"Aligned rows: {aligned_data.shape[0]}")
    print(f"Aligned columns: {aligned_data.shape[1]}")