import os
import pandas as pd


class TemporalFeatureEngineer:
    """
    Creates temporal lag, rolling, and future-target features
    for AirTwinNet air-quality forecasting.
    """

    def __init__(self):
        self.input_file = "data/raw/air_quality_sample.csv"

        self.output_dir = "data/processed"

        self.output_file = os.path.join(
            self.output_dir,
            "air_quality_temporal.csv"
        )

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

    def load_data(self):
        """Load and prepare raw air-quality data."""

        if not os.path.exists(self.input_file):
            raise FileNotFoundError(
                f"Input file not found: {self.input_file}"
            )

        df = pd.read_csv(self.input_file)

        required_columns = [
            "timestamp",
            "city",
            "temperature",
            "humidity",
            "pm2_5",
            "pm10",
            "no2",
            "co",
            "o3"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["timestamp"]
        )

        df = df.sort_values(
            ["city", "timestamp"]
        ).reset_index(
            drop=True
        )

        return df

    def create_features(self, df):
        """Create lag, rolling and future target features."""

        # --------------------------------------------------
        # Calendar features
        # --------------------------------------------------

        df["hour"] = df["timestamp"].dt.hour

        df["day_of_week"] = (
            df["timestamp"].dt.dayofweek
        )

        df["month"] = (
            df["timestamp"].dt.month
        )

        df["day"] = (
            df["timestamp"].dt.day
        )

        df["week_of_year"] = (
            df["timestamp"].dt.isocalendar().week
        ).astype(int)

        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype(int)

        # --------------------------------------------------
        # PM2.5 lag features
        # --------------------------------------------------

        df["pm2_5_lag_1"] = (
            df.groupby("city")["pm2_5"]
            .shift(1)
        )

        df["pm2_5_lag_3"] = (
            df.groupby("city")["pm2_5"]
            .shift(3)
        )

        df["pm2_5_lag_6"] = (
            df.groupby("city")["pm2_5"]
            .shift(6)
        )

        df["pm2_5_lag_24"] = (
            df.groupby("city")["pm2_5"]
            .shift(24)
        )

        # --------------------------------------------------
        # PM10 lag features
        # --------------------------------------------------

        df["pm10_lag_1"] = (
            df.groupby("city")["pm10"]
            .shift(1)
        )

        df["pm10_lag_3"] = (
            df.groupby("city")["pm10"]
            .shift(3)
        )

        df["pm10_lag_6"] = (
            df.groupby("city")["pm10"]
            .shift(6)
        )

        df["pm10_lag_24"] = (
            df.groupby("city")["pm10"]
            .shift(24)
        )

        # --------------------------------------------------
        # Rolling PM2.5 features
        # --------------------------------------------------

        df["pm2_5_rolling_6"] = (
            df.groupby("city")["pm2_5"]
            .transform(
                lambda x: x.shift(1)
                .rolling(window=6)
                .mean()
            )
        )

        df["pm2_5_rolling_24"] = (
            df.groupby("city")["pm2_5"]
            .transform(
                lambda x: x.shift(1)
                .rolling(window=24)
                .mean()
            )
        )

        # --------------------------------------------------
        # Rolling PM10 features
        # --------------------------------------------------

        df["pm10_rolling_6"] = (
            df.groupby("city")["pm10"]
            .transform(
                lambda x: x.shift(1)
                .rolling(window=6)
                .mean()
            )
        )

        df["pm10_rolling_24"] = (
            df.groupby("city")["pm10"]
            .transform(
                lambda x: x.shift(1)
                .rolling(window=24)
                .mean()
            )
        )

        # --------------------------------------------------
        # Previous pollutant values
        # --------------------------------------------------

        df["no2_lag_1"] = (
            df.groupby("city")["no2"]
            .shift(1)
        )

        df["co_lag_1"] = (
            df.groupby("city")["co"]
            .shift(1)
        )

        df["o3_lag_1"] = (
            df.groupby("city")["o3"]
            .shift(1)
        )

        # --------------------------------------------------
        # Future targets
        # --------------------------------------------------

        df["pm2_5_future"] = (
            df.groupby("city")["pm2_5"]
            .shift(-1)
        )

        df["pm10_future"] = (
            df.groupby("city")["pm10"]
            .shift(-1)
        )

        return df

    def clean_data(self, df):
        """Remove rows where temporal features are unavailable."""

        feature_columns = [
            "pm2_5_lag_1",
            "pm2_5_lag_3",
            "pm2_5_lag_6",
            "pm2_5_lag_24",
            "pm10_lag_1",
            "pm10_lag_3",
            "pm10_lag_6",
            "pm10_lag_24",
            "pm2_5_rolling_6",
            "pm2_5_rolling_24",
            "pm10_rolling_6",
            "pm10_rolling_24",
            "no2_lag_1",
            "co_lag_1",
            "o3_lag_1",
            "pm2_5_future",
            "pm10_future"
        ]

        df = df.dropna(
            subset=feature_columns
        ).reset_index(
            drop=True
        )

        return df

    def save_data(self, df):
        """Save processed temporal dataset."""

        df.to_csv(
            self.output_file,
            index=False
        )

        print(
            f"\nSaved temporal dataset:"
        )

        print(
            self.output_file
        )

    def run(self):
        """Run complete temporal feature engineering."""

        print("\n" + "=" * 60)
        print("AirTwinNet Temporal Feature Engineering")
        print("=" * 60)

        df = self.load_data()

        print(
            f"\nOriginal shape: {df.shape}"
        )

        df = self.create_features(df)

        print(
            f"After feature creation: {df.shape}"
        )

        df = self.clean_data(df)

        print(
            f"After temporal cleaning: {df.shape}"
        )

        self.save_data(df)

        print(
            "\nTemporal features created successfully."
        )

        print(
            "\nNew forecasting columns:"
        )

        print(
            "PM2.5 lag features"
        )

        print(
            "PM10 lag features"
        )

        print(
            "6-hour and 24-hour rolling features"
        )

        print(
            "Previous pollutant features"
        )

        print(
            "Future PM2.5 target"
        )

        print(
            "Future PM10 target"
        )

        print("\n" + "=" * 60)
        print("TEMPORAL FEATURE ENGINEERING COMPLETE")
        print("=" * 60)


if __name__ == "__main__":

    engineer = TemporalFeatureEngineer()

    engineer.run()    