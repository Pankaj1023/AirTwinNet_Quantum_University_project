from src.airtwinnet.components.dashboard import create_dashboard_app


app = create_dashboard_app()


if __name__ == "__main__":
    print("Starting AirTwinNet Dashboard...")
    app.run(debug=False)