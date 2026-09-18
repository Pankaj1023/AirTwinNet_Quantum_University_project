from src.airtwinnet.components.temporal_prediction_service import TemporalPredictionService


def test_temporal_prediction_service():
    service = TemporalPredictionService()

    assert service.pm25_model is not None
    assert service.pm10_model is not None
    assert len(service.features) == 26