import pandas as pd


class UrbanAirQualityDigitalTwin:

    def __init__(self, fused_data: pd.DataFrame):

        self.state = fused_data.copy()

    def update_state(self, new_data: pd.DataFrame):

        if new_data is None or new_data.empty:
            raise ValueError(
                "Digital Twin cannot be updated with empty data."
            )

        self.state = new_data.copy()

        print("Digital Twin state updated successfully.")

        return self.state

    def get_current_state(self):

        if self.state.empty:
            raise ValueError(
                "Digital Twin state is empty."
            )

        latest_timestamp = self.state["timestamp"].max()

        current_state = self.state[
            self.state["timestamp"] == latest_timestamp
        ].copy()

        return current_state

    def get_city_state(self, city: str):

        if "city" not in self.state.columns:
            raise ValueError(
                "City information is required."
            )

        city_state = self.state[
            self.state["city"].str.lower() == city.lower()
        ].copy()

        return city_state

    def simulate_state(self, updates: dict):

        simulated_state = self.state.copy()

        for column, value in updates.items():

            if column in simulated_state.columns:
                simulated_state[column] = value

        return simulated_state