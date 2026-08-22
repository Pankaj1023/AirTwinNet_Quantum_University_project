import pandas as pd

from src.airtwinnet.components.digital_twin import (
    UrbanAirQualityDigitalTwin
)


def test_digital_twin():

    data_path = "data/raw/air_quality_sample.csv"

    data = pd.read_csv(data_path)

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    digital_twin = UrbanAirQualityDigitalTwin(data)

    # Test current state
    current_state = digital_twin.get_current_state()

    assert current_state is not None
    assert not current_state.empty

    # Test city state
    city_state = digital_twin.get_city_state("Roorkee")

    assert city_state is not None
    assert not city_state.empty

    # Test state update
    updated_state = digital_twin.update_state(data)

    assert updated_state is not None
    assert not updated_state.empty

    # Test what-if simulation
    simulated_state = digital_twin.simulate_state(
        {"temperature": 35.0}
    )

    assert simulated_state is not None
    assert not simulated_state.empty

    assert "temperature" in simulated_state.columns

    print("Digital Twin test passed successfully.")
    print(f"Current state rows: {current_state.shape[0]}")
    print(f"City state rows: {city_state.shape[0]}")
    print(f"Digital Twin rows: {digital_twin.state.shape[0]}")