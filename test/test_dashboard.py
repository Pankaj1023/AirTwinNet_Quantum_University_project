from src.airtwinnet.components.dashboard import (
    create_dashboard_app
)


def test_dashboard():

    app = create_dashboard_app()

    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200

    data = response.get_json()

    assert data["project"] == "AirTwinNet"

    assert data["module"] == (
        "Decision-Support Dashboard"
    )

    assert data["status"] == "running"

    dashboard_response = client.get(
        "/api/dashboard"
    )

    assert dashboard_response.status_code == 200

    dashboard_data = dashboard_response.get_json()

    assert "pm2_5" in dashboard_data
    assert "pm10" in dashboard_data
    assert "aqi" in dashboard_data
    assert "confidence" in dashboard_data

    print("Dashboard test passed successfully.")