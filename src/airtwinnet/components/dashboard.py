from flask import Flask, jsonify


def create_dashboard_app():

    app = Flask(__name__)

    dashboard_state = {
        "pm2_5": None,
        "pm10": None,
        "aqi": None,
        "confidence": None,
        "explanation": None
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