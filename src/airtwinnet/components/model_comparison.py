import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class AirQualityModelComparison:

    def __init__(self, data_path):

        self.data_path = data_path

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

    def load_data(self):

        data = pd.read_csv(
            self.data_path
        )

        data["timestamp"] = pd.to_datetime(
            data["timestamp"]
        )

        data = data.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        data["hour"] = data["timestamp"].dt.hour

        data["day_of_week"] = (
            data["timestamp"].dt.dayofweek
        )

        data["month"] = (
            data["timestamp"].dt.month
        )

        return data

    def compare_models(self):

        print(
            "Starting AirTwinNet model comparison..."
        )

        data = self.load_data()

        X = data[self.feature_columns]

        y = data[self.target_columns]

        split_index = int(
            len(data) * 0.8
        )

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        models = {

            "Random Forest": MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
            ),

            "Gradient Boosting": MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=42
                )
            )
        }

        results = []

        for model_name, model in models.items():

            print(
                f"\nTraining {model_name}..."
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            mse = mean_squared_error(
                y_test,
                predictions
            )

            rmse = mse ** 0.5

            r2 = r2_score(
                y_test,
                predictions
            )

            results.append({
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

            print(
                f"{model_name} completed."
            )

            print(
                "MAE:",
                mae
            )

            print(
                "RMSE:",
                rmse
            )

            print(
                "R2:",
                r2
            )

        results_df = pd.DataFrame(
            results
        )

        results_df = results_df.sort_values(
            "R2",
            ascending=False
        ).reset_index(drop=True)

        print(
            "\n========== MODEL COMPARISON =========="
        )

        print(
            results_df.to_string(
                index=False
            )
        )

        best_model = results_df.iloc[0]

        print(
            "\nBest Model:",
            best_model["Model"]
        )

        print(
            "Best R2:",
            best_model["R2"]
        )

        return results_df


if __name__ == "__main__":

    comparison = AirQualityModelComparison(
        "data/raw/air_quality_sample.csv"
    )

    comparison.compare_models()