from flask import Flask, jsonify
import os
import pandas as pd

from src.airtwinnet.components.prediction_service import (
    AirQualityPredictionService
)


CURRENT_DATA_PATH = "data/raw/air_quality_sample.csv"

HYBRID_RESULTS_PATH = (
    "artifacts/hybrid_future_test_predictions.csv"
)

HYBRID_EVALUATION_PATH = (
    "artifacts/hybrid_future_evaluation_results.csv"
)

HYBRID_CITY_RESULTS_PATH = (
    "artifacts/hybrid_future_city_wise_results.csv"
)


def create_dashboard_app():

    app = Flask(__name__)

    # =================================================
    # CURRENT AIR QUALITY PREDICTION
    # =================================================

    prediction_service = AirQualityPredictionService(
        "artifacts/air_quality_model.pkl"
    )

    # =================================================
    # LOAD CURRENT DATA
    # =================================================

    if not os.path.exists(CURRENT_DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {CURRENT_DATA_PATH}"
        )

    data = pd.read_csv(
        CURRENT_DATA_PATH
    )

    latest_data = data.tail(1)

    # =================================================
    # CURRENT PREDICTION
    # =================================================

    prediction = prediction_service.predict(
        latest_data
    )

    # =================================================
    # DASHBOARD STATE
    # =================================================

    dashboard_state = {

        # ---------------------------------------------
        # Current Air Quality
        # ---------------------------------------------

        "pm2_5": prediction["pm2_5"],

        "pm10": prediction["pm10"],

        "aqi": prediction["aqi"],

        "category": prediction["category"],

        # ---------------------------------------------
        # Confidence
        # ---------------------------------------------

        "confidence": prediction["confidence"],

        # ---------------------------------------------
        # XAI
        # ---------------------------------------------

        "explanation": prediction["explanation"],

        "pm2_5_dominant_feature": (
            prediction[
                "pm2_5_dominant_feature"
            ]
        ),

        "pm2_5_feature_importance": (
            prediction[
                "pm2_5_feature_importance"
            ]
        ),

        "pm10_dominant_feature": (
            prediction[
                "pm10_dominant_feature"
            ]
        ),

        "pm10_feature_importance": (
            prediction[
                "pm10_feature_importance"
            ]
        ),

        # ---------------------------------------------
        # Model Information
        # ---------------------------------------------

        "pm2_5_model": (
            prediction["pm2_5_model"]
        ),

        "pm10_model": (
            prediction["pm10_model"]
        ),

        # ---------------------------------------------
        # Environmental Conditions
        # ---------------------------------------------

        "temperature": float(
            latest_data.iloc[0][
                "temperature"
            ]
        ),

        "humidity": float(
            latest_data.iloc[0][
                "humidity"
            ]
        ),

        "no2": float(
            latest_data.iloc[0][
                "no2"
            ]
        ),

        "co": float(
            latest_data.iloc[0][
                "co"
            ]
        ),

        "o3": float(
            latest_data.iloc[0][
                "o3"
            ]
        ),

        # ---------------------------------------------
        # Location / Timestamp
        # ---------------------------------------------

        "city": str(
            latest_data.iloc[0][
                "city"
            ]
        ),

        "timestamp": str(
            latest_data.iloc[0][
                "timestamp"
            ]
        ),

        # ---------------------------------------------
        # System Status
        # ---------------------------------------------

        "status": (
            "prediction_available"
        )
    }

    # =================================================
    # LOAD HYBRID FUTURE RESULTS
    # =================================================

    hybrid_results = None
    hybrid_evaluation = None
    hybrid_city_results = None

    if os.path.exists(
        HYBRID_RESULTS_PATH
    ):

        hybrid_results = pd.read_csv(
            HYBRID_RESULTS_PATH
        )

        hybrid_results[
            "timestamp"
        ] = pd.to_datetime(
            hybrid_results[
                "timestamp"
            ]
        )

    if os.path.exists(
        HYBRID_EVALUATION_PATH
    ):

        hybrid_evaluation = pd.read_csv(
            HYBRID_EVALUATION_PATH
        )

    if os.path.exists(
        HYBRID_CITY_RESULTS_PATH
    ):

        hybrid_city_results = pd.read_csv(
            HYBRID_CITY_RESULTS_PATH
        )

    # =================================================
    # HOME ROUTE
    # =================================================

    @app.route("/")
    def home():

        return jsonify({

            "project": "AirTwinNet",

            "module": (
                "Decision-Support Dashboard"
            ),

            "status": "running",

            "future_forecasting": (
                "available"
                if hybrid_results is not None
                else "not_available"
            )

        })

    # =================================================
    # CURRENT DASHBOARD API
    # =================================================

    @app.route("/api/dashboard")
    def dashboard():

        return jsonify(
            dashboard_state
        )

    # =================================================
    # FUTURE HYBRID FORECAST API
    # =================================================

    @app.route("/api/future-forecast")
    def future_forecast():

        if hybrid_results is None:

            return jsonify({

                "status": "not_available",

                "message": (
                    "Hybrid future prediction "
                    "results are not available."
                )

            }), 404

        # Convert timestamps to strings
        forecast_data = (
            hybrid_results.copy()
        )

        forecast_data[
            "timestamp"
        ] = forecast_data[
            "timestamp"
        ].astype(str)

        return jsonify({

            "status": "available",

            "model": "Hybrid",

            "ml_weight": 0.60,

            "st_gnn_weight": 0.40,

            "records": (
                forecast_data
                .to_dict(orient="records")
            )

        })

    # =================================================
    # HYBRID MODEL EVALUATION API
    # =================================================

    @app.route("/api/hybrid-evaluation")
    def hybrid_evaluation_api():

        if hybrid_evaluation is None:

            return jsonify({

                "status": "not_available",

                "message": (
                    "Hybrid evaluation results "
                    "are not available."
                )

            }), 404

        return jsonify({

            "status": "available",

            "results": (
                hybrid_evaluation
                .to_dict(orient="records")
            )

        })

    # =================================================
    # CITY-WISE HYBRID API
    # =================================================

    @app.route("/api/hybrid-city-results")
    def hybrid_city_api():

        if hybrid_city_results is None:

            return jsonify({

                "status": "not_available",

                "message": (
                    "City-wise hybrid results "
                    "are not available."
                )

            }), 404

        return jsonify({

            "status": "available",

            "results": (
                hybrid_city_results
                .to_dict(orient="records")
            )

        })

    # =================================================
    # HEALTH CHECK
    # =================================================

    @app.route("/health")
    def health():

        return jsonify({

            "status": "healthy",

            "prediction_service": "active",

            "current_model": (
                "GradientBoosting"
            ),

            "future_model": (
                "Temporal ML + ST-GNN + Hybrid"
            ),

            "future_forecasting": (
                "active"
                if hybrid_results is not None
                else "not_available"
            )

        })

    return app