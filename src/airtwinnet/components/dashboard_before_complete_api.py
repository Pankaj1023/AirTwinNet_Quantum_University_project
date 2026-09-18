from flask import Flask, jsonify

from src.airtwinnet.components.prediction_service import (
    AirQualityPredictionService
)

import pandas as pd


def create_dashboard_app():

    app = Flask(__name__)

    # Load trained model
    prediction_service = AirQualityPredictionService(
        "artifacts/air_quality_model.pkl"
    )

    # Load current air-quality data
    data = pd.read_csv(
        "data/raw/air_quality_sample.csv"
    )

    latest_data = data.tail(1)

    # Generate prediction
    prediction = prediction_service.predict(
        latest_data
    )

    dashboard_state = {
        "pm2_5": prediction["pm2_5"],
        "pm10": prediction["pm10"],
        "aqi": prediction["aqi"],
        "confidence": None,
        "explanation": None,
        "category": prediction["category"],
        "status": "prediction_available"
    }

    @app.route("/")
    def home():

        return jsonify({
            "project": "AirTwinNet",
            "module": "Decision-Support Dashboard",
            "status": "running"
        })

    @app.route("/api/dashboard")
    def dashboard():

        return jsonify(dashboard_state)

    return app