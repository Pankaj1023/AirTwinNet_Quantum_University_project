import pandas as pd


class DataPreprocessing:

    def __init__(self, data: pd.DataFrame):

        self.data = data.copy()

    def initiate_data_preprocessing(self):

        print("Starting data preprocessing...")

        # 1. Remove duplicate records
        self.data = self.data.drop_duplicates()

        # 2. Convert timestamp column
        if "timestamp" in self.data.columns:
            self.data["timestamp"] = pd.to_datetime(
                self.data["timestamp"],
                errors="coerce"
            )

        # 3. Convert numeric columns
        numeric_columns = [
            "temperature",
            "humidity",
            "pm2_5",
            "pm10",
            "no2",
            "co",
            "o3"
        ]

        for column in numeric_columns:
            if column in self.data.columns:
                self.data[column] = pd.to_numeric(
                    self.data[column],
                    errors="coerce"
                )

        # 4. Handle missing values
        for column in numeric_columns:
            if column in self.data.columns:
                if self.data[column].isnull().any():
                    self.data[column] = self.data[column].fillna(
                        self.data[column].median()
                    )

        # 5. Remove rows with invalid timestamp
        if "timestamp" in self.data.columns:
            self.data = self.data.dropna(subset=["timestamp"])

        print("Data preprocessing completed successfully.")
        print(f"Processed dataset shape: {self.data.shape}")

        return self.data