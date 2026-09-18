import pandas as pd


class AirQualityFeatureEngineer:

    def __init__(self):

        self.feature_columns = [
            "temperature",
            "humidity",
            "no2",
            "co",
            "o3",
            "hour",
            "day_of_week",
            "month"
        ]

        self.target_columns = [
            "pm2_5",
            "pm10"
        ]

    def transform(self, data: pd.DataFrame):

        if data is None or data.empty:
            raise ValueError(
                "Input data cannot be empty."
            )

        data = data.copy()

        # Convert timestamp
        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce"
        )

        data = data.dropna(
            subset=["timestamp"]
        )

        # Temporal features
        data["hour"] = data["timestamp"].dt.hour

        data["day_of_week"] = (
            data["timestamp"].dt.dayofweek
        )

        data["month"] = (
            data["timestamp"].dt.month
        )

        # Clean city names
        data["city"] = (
            data["city"]
            .astype(str)
            .str.strip()
        )

        required_columns = (
            self.feature_columns
            + self.target_columns
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        print(
            "Feature engineering completed successfully."
        )

        print(
            "Feature columns:",
            self.feature_columns
        )

        print(
            "Target columns:",
            self.target_columns
        )

        return data


if __name__ == "__main__":

    data_path = (
        "data/raw/air_quality_sample.csv"
    )

    data = pd.read_csv(data_path)

    engineer = AirQualityFeatureEngineer()

    transformed_data = engineer.transform(
        data
    )

    print("\nTransformed Shape:")
    print(transformed_data.shape)

    print("\nSample:")
    print(
        transformed_data[
            [
                "timestamp",
                "city",
                "hour",
                "day_of_week",
                "month",
                "temperature",
                "humidity",
                "no2",
                "co",
                "o3",
                "pm2_5",
                "pm10"
            ]
        ].head(10).to_string(index=False)
    )