from flask import Flask, jsonify, request


def create_dashboard_app():

    app = Flask(__name__)

    dashboard_state = {
        "pm2_5": None,
        "pm10": None,
        "aqi": None,
        "confidence": None,
        "explanation": None,
        "status": "waiting_for_prediction"
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

    @app.route("/api/prediction", methods=["POST"])
    def update_prediction():

        data = request.get_json()

        if data is None:
            return jsonify({
                "status": "error",
                "message": "JSON prediction data is required."
            }), 400

        dashboard_state["pm2_5"] = data.get("pm2_5")
        dashboard_state["pm10"] = data.get("pm10")
        dashboard_state["aqi"] = data.get("aqi")
        dashboard_state["confidence"] = data.get("confidence")
        dashboard_state["explanation"] = data.get("explanation")
        dashboard_state["status"] = "prediction_available"

        return jsonify({
            "status": "success",
            "message": "Prediction updated successfully.",
            "dashboard_state": dashboard_state
        })

    return app
