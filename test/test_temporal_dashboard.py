from src.airtwinnet.components.temporal_dashboard import TemporalDashboard


def test_temporal_dashboard():
    dashboard = TemporalDashboard()

    forecast = dashboard.get_forecast(5)

    assert len(forecast) == 5
    assert "predicted_pm2_5" in forecast.columns
    assert "predicted_pm10" in forecast.columns
    assert forecast["predicted_pm2_5"].notna().all()
    assert forecast["predicted_pm10"].notna().all()