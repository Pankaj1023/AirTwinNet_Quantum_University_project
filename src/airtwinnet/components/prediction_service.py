import pickle
import pandas as pd

from src.airtwinnet.components.aqi_calculator import AQICalculator
from src.airtwinnet.components.explainable_ai import ExplainableAI


class AirQualityPredictionService:

    def __init__(self, model_path):

        with open(model_path, "rb") as file:
            self.model_package = pickle.load(file)

        self.pm25_model = self.model_package["pm25_model"]
        self.pm10_model = self.model_package["pm10_model"]

        self.feature_columns = self.model_package["features"]

        self.pm25_model_name = self.model_package[
            "pm25_model_name"
        ]

        self.pm10_model_name = self.model_package[
            "pm10_model_name"
        ]

        self.aqi_calculator = AQICalculator()

        self.xai_features = self.feature_columns

        self.xai = ExplainableAI(
            self.xai_features
        )

    def create_features(self, input_data):

        df = input_data.copy()

        if "timestamp" not in df.columns:
            raise ValueError(
                "Timestamp column is required for feature engineering."
            )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        if df["timestamp"].isna().any():
            raise ValueError(
                "Invalid timestamp found in input data."
            )

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
            df["timestamp"]
            .dt.isocalendar()
            .week
            .astype(int)
        )

        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype(int)

        df["temp_humidity"] = (
            df["temperature"] *
            df["humidity"]
        )

        df["no2_co"] = (
            df["no2"] *
            df["co"]
        )

        df["no2_o3"] = (
            df["no2"] *
            df["o3"]
        )

        df["co_o3"] = (
            df["co"] *
            df["o3"]
        )

        X = df[
            self.feature_columns
        ]

        return X

    def predict(self, input_data):

        if input_data is None or input_data.empty:
            raise ValueError(
                "Input data cannot be empty."
            )

        X = self.create_features(
            input_data
        )

        # PM2.5 prediction
        pm25_prediction = (
            self.pm25_model.predict(X)
        )

        # PM10 prediction
        pm10_prediction = (
            self.pm10_model.predict(X)
        )

        pm25 = float(
            pm25_prediction[0]
        )

        pm10 = float(
            pm10_prediction[0]
        )

        # AQI calculation
        aqi = self.aqi_calculator.calculate_aqi(
            pm25,
            pm10
        )

        category = (
            self.aqi_calculator.get_category(
                aqi
            )
        )

        # ------------------------------------------------
        # Explainable AI
        # ------------------------------------------------

        try:

            model_explanation = (
                self.xai.explain_separate_models(
                    self.pm25_model,
                    self.pm10_model,
                    self.feature_columns
                )
            )

            pm25_dominant_feature = (
                model_explanation[
                    "pm2_5"
                ]["dominant_feature"]
            )

            pm25_importance = (
                model_explanation[
                    "pm2_5"
                ]["dominant_importance"]
            )

            pm10_dominant_feature = (
                model_explanation[
                    "pm10"
                ]["dominant_feature"]
            )

            pm10_importance = (
                model_explanation[
                    "pm10"
                ]["dominant_importance"]
            )

            explanation = (
                f"PM2.5 prediction is mainly "
                f"influenced by "
                f"{pm25_dominant_feature}, "
                f"while PM10 prediction is mainly "
                f"influenced by "
                f"{pm10_dominant_feature}."
            )

        except Exception as e:

            explanation = (
                "Prediction generated successfully "
                "using the improved Gradient "
                "Boosting models."
            )

            pm25_dominant_feature = "N/A"
            pm25_importance = 0.0

            pm10_dominant_feature = "N/A"
            pm10_importance = 0.0

        # ------------------------------------------------
        # Confidence
        # ------------------------------------------------

        confidence_values = []

        if "pm2_5" in input_data.columns:

            actual_pm25 = float(
                input_data.iloc[0]["pm2_5"]
            )

            pm25_confidence = (
                1.0 -
                abs(pm25 - actual_pm25) /
                max(abs(actual_pm25), 1.0)
            )

            confidence_values.append(
                pm25_confidence
            )

        if "pm10" in input_data.columns:

            actual_pm10 = float(
                input_data.iloc[0]["pm10"]
            )

            pm10_confidence = (
                1.0 -
                abs(pm10 - actual_pm10) /
                max(abs(actual_pm10), 1.0)
            )

            confidence_values.append(
                pm10_confidence
            )

        if confidence_values:

            confidence = (
                sum(confidence_values) /
                len(confidence_values)
            )

        else:

            confidence = 0.0

        confidence = max(
            0.0,
            min(1.0, confidence)
        )

        # ------------------------------------------------
        # Final result
        # ------------------------------------------------

        return {

            "pm2_5": round(
                pm25,
                2
            ),

            "pm10": round(
                pm10,
                2
            ),

            "aqi": round(
                float(aqi),
                2
            ),

            "category": category,

            "confidence": round(
                confidence,
                4
            ),

            "explanation": explanation,

            "pm2_5_dominant_feature":
                pm25_dominant_feature,

            "pm2_5_feature_importance":
                round(
                    float(pm25_importance),
                    4
                ),

            "pm10_dominant_feature":
                pm10_dominant_feature,

            "pm10_feature_importance":
                round(
                    float(pm10_importance),
                    4
                ),

            "pm2_5_model":
                self.pm25_model_name,

            "pm10_model":
                self.pm10_model_name
        }


if __name__ == "__main__":

    service = AirQualityPredictionService(
        "artifacts/air_quality_model.pkl"
    )

    data = pd.read_csv(
        "data/raw/air_quality_sample.csv"
    )

    latest_data = data.tail(1)

    result = service.predict(
        latest_data
    )

    print(
        "\nAirTwinNet Complete Prediction"
    )

    print(
        "--------------------------------"
    )

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )