import pandas as pd

from src.airtwinnet.components.data_fusion import (
    MultiModalDataFusion
)


def test_data_fusion():

    data_path = "data/raw/air_quality_sample.csv"

    data = pd.read_csv(data_path)

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    fusion = MultiModalDataFusion(data)

    fused_data = fusion.initiate_data_fusion()

    assert fused_data is not None
    assert not fused_data.empty

    # Check common spatial-temporal identifiers
    assert "timestamp" in fused_data.columns
    assert "city" in fused_data.columns

    # Check important air-quality/weather features
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
        assert feature in fused_data.columns

    print("Multi-modal data fusion test passed successfully.")
    print(f"Fused rows: {fused_data.shape[0]}")
    print(f"Fused columns: {fused_data.shape[1]}")