
import pandas as pd
import streamlit as st
import numpy as np

from src.airtwinnet.components.prediction_service import (
    AirQualityPredictionService
)

from src.airtwinnet.components.aqi_calculator import (
    AQICalculator
)

from src.airtwinnet.components.temporal_dashboard import (
    TemporalDashboard
)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AirTwinNet Dashboard",
    page_icon="🌍",
    layout="wide"
)


# --------------------------------------------------
# Load Services
# --------------------------------------------------

@st.cache_resource
def load_prediction_service():

    return AirQualityPredictionService(
        "artifacts/air_quality_model.pkl"
    )


@st.cache_resource
def load_temporal_dashboard():

    return TemporalDashboard(
        "data/processed/air_quality_temporal.csv",
        "artifacts/temporal_air_quality_model.pkl"
    )


@st.cache_data
def load_data():

    return pd.read_csv(
        "data/raw/air_quality_sample.csv"
    )


service = load_prediction_service()
temporal_dashboard = load_temporal_dashboard()
data = load_data()

aqi_calculator = AQICalculator()


# --------------------------------------------------
# Prepare Data
# --------------------------------------------------

data["timestamp"] = pd.to_datetime(
    data["timestamp"]
)

data = data.sort_values(
    "timestamp"
).reset_index(drop=True)


# --------------------------------------------------
# Environmental Features
# --------------------------------------------------

environmental_features = [
    "temperature",
    "humidity",
    "no2",
    "co",
    "o3"
]


# --------------------------------------------------
# Generate Historical Predictions
# --------------------------------------------------

model_features = service.create_features(data)

data["predicted_pm2_5"] = (
    service.pm25_model.predict(model_features)
)

data["predicted_pm10"] = (
    service.pm10_model.predict(model_features)
)


# --------------------------------------------------
# Calculate Historical AQI
# --------------------------------------------------

data["predicted_aqi"] = data.apply(
    lambda row: aqi_calculator.calculate_aqi(
        row["predicted_pm2_5"],
        row["predicted_pm10"]
    ),
    axis=1
)

data["aqi_category"] = data[
    "predicted_aqi"
].apply(
    aqi_calculator.get_category
)


# --------------------------------------------------
# Latest Prediction
# --------------------------------------------------

latest_data = data.tail(1)

latest_prediction = service.predict(
    latest_data
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🌍 AirTwinNet")

st.subheader(
    "Digital Twin Enabled Urban Air Quality "
    "Prediction & Decision-Support Dashboard"
)

st.divider()


# --------------------------------------------------
# Location Information
# --------------------------------------------------

city = latest_data.iloc[0]["city"]

timestamp = latest_data.iloc[0]["timestamp"]

st.write(
    f"**Location:** {city}  |  "
    f"**Latest Observation:** {timestamp}"
)


# --------------------------------------------------
# Main Metrics
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "PM2.5",
        f"{latest_prediction['pm2_5']:.2f}"
    )


with col2:

    st.metric(
        "PM10",
        f"{latest_prediction['pm10']:.2f}"
    )


with col3:

    st.metric(
        "AQI",
        f"{latest_prediction['aqi']:.2f}"
    )


with col4:

    st.metric(
        "Confidence",
        f"{latest_prediction['confidence'] * 100:.2f}%"
    )


st.divider()


# --------------------------------------------------
# AQI Status
# --------------------------------------------------

st.subheader(
    "🌫️ Air Quality Status"
)

st.info(
    f"AQI: {latest_prediction['aqi']:.2f}  |  "
    f"Category: {latest_prediction['category']}"
)


# --------------------------------------------------
# Historical Air Quality Trends
# --------------------------------------------------

st.subheader(
    "📈 Historical Air Quality Trends"
)

trend_data = data[
    [
        "timestamp",
        "predicted_pm2_5",
        "predicted_pm10"
    ]
].copy()

trend_data = trend_data.set_index(
    "timestamp"
)

trend_data.columns = [
    "PM2.5",
    "PM10"
]

# Downsample for faster dashboard rendering
step = max(1, len(trend_data) // 2000)

trend_data = trend_data.iloc[::step]

st.line_chart(
    trend_data,
    width="stretch"
)


# --------------------------------------------------
# Actual vs Predicted
# --------------------------------------------------

st.subheader(
    "🎯 Actual vs Predicted"
)

comparison = data[
    [
        "timestamp",
        "pm2_5",
        "predicted_pm2_5",
        "pm10",
        "predicted_pm10"
    ]
].copy()

comparison = comparison.set_index(
    "timestamp"
)

comparison.columns = [
    "Actual PM2.5",
    "Predicted PM2.5",
    "Actual PM10",
    "Predicted PM10"
]

# Downsample for faster rendering
step = max(1, len(comparison) // 2000)

comparison = comparison.iloc[::step]

st.line_chart(
    comparison,
    width="stretch"
)


# --------------------------------------------------
# Historical AQI Trend
# --------------------------------------------------

st.subheader(
    "📊 Historical AQI Trend"
)

aqi_trend = data[
    [
        "timestamp",
        "predicted_aqi"
    ]
].copy()

aqi_trend = aqi_trend.set_index(
    "timestamp"
)

aqi_trend.columns = [
    "Predicted AQI"
]

# Downsample for faster dashboard rendering
step = max(1, len(aqi_trend) // 2000)

aqi_trend = aqi_trend.iloc[::step]

st.line_chart(
    aqi_trend,
    width="stretch"
)


# ==================================================
# TEMPORAL FUTURE FORECAST
# ==================================================

st.divider()

st.subheader(
    "🔮 Future Air Quality Forecast"
)

st.write(
    "Temporal forecasting model based on historical "
    "lag and rolling features."
)


# Generate future forecast
forecast = temporal_dashboard.get_forecast(10)


# Forecast metrics
forecast_col1, forecast_col2 = st.columns(2)


with forecast_col1:

    st.metric(
        "Latest Predicted Future PM2.5",
        f"{forecast['predicted_pm2_5'].iloc[-1]:.2f}"
    )


with forecast_col2:

    st.metric(
        "Latest Predicted Future PM10",
        f"{forecast['predicted_pm10'].iloc[-1]:.2f}"
    )


# --------------------------------------------------
# Future PM2.5 Forecast Chart
# --------------------------------------------------

st.write(
    "### Future PM2.5 Forecast"
)

pm25_forecast = forecast[
    [
        "timestamp",
        "predicted_pm2_5"
    ]
].copy()

pm25_forecast = pm25_forecast.set_index(
    "timestamp"
)

pm25_forecast.columns = [
    "Predicted Future PM2.5"
]

st.line_chart(
    pm25_forecast,
    width="stretch"
)


# --------------------------------------------------
# Future PM10 Forecast Chart
# --------------------------------------------------

st.write(
    "### Future PM10 Forecast"
)

pm10_forecast = forecast[
    [
        "timestamp",
        "predicted_pm10"
    ]
].copy()

pm10_forecast = pm10_forecast.set_index(
    "timestamp"
)

pm10_forecast.columns = [
    "Predicted Future PM10"
]

st.line_chart(
    pm10_forecast,
    width="stretch"
)


# --------------------------------------------------
# Actual vs Future Predicted
# --------------------------------------------------

st.write(
    "### Actual vs Future Predicted"
)

future_comparison = forecast[
    [
        "timestamp",
        "pm2_5_future",
        "predicted_pm2_5",
        "pm10_future",
        "predicted_pm10"
    ]
].copy()

future_comparison = future_comparison.set_index(
    "timestamp"
)

future_comparison.columns = [
    "Actual Future PM2.5",
    "Predicted Future PM2.5",
    "Actual Future PM10",
    "Predicted Future PM10"
]

st.line_chart(
    future_comparison,
    width="stretch"
)


# --------------------------------------------------
# Forecast Table
# --------------------------------------------------

st.write(
    "### Forecast Details"
)

forecast_table = forecast.copy()

forecast_table["pm2_5_future"] = (
    forecast_table["pm2_5_future"].round(2)
)

forecast_table["pm10_future"] = (
    forecast_table["pm10_future"].round(2)
)

forecast_table["predicted_pm2_5"] = (
    forecast_table["predicted_pm2_5"].round(2)
)

forecast_table["predicted_pm10"] = (
    forecast_table["predicted_pm10"].round(2)
)

st.dataframe(
    forecast_table,
    width="stretch",
    hide_index=True
)


# ==================================================
# Explainable AI
# ==================================================

st.subheader(
    "🧠 Explainable AI"
)

st.info(
    latest_prediction["explanation"]
)


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "PM2.5 Dominant Feature",
        latest_prediction[
            "pm2_5_dominant_feature"
        ]
    )

    st.write(
        "Feature Importance: "
        f"{latest_prediction['pm2_5_feature_importance'] * 100:.2f}%"
    )


with col2:

    st.metric(
        "PM10 Dominant Feature",
        latest_prediction[
            "pm10_dominant_feature"
        ]
    )

    st.write(
        "Feature Importance: "
        f"{latest_prediction['pm10_feature_importance'] * 100:.2f}%"
    )


# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

st.subheader(
    "🔍 Feature Importance"
)

xai_details = service.xai.explain_separate_models(
    service.pm25_model,
    service.pm10_model,
    service.feature_columns
)


pm25_importance = pd.DataFrame(
    xai_details["pm2_5"][
        "feature_importance"
    ].items(),
    columns=[
        "Feature",
        "Importance"
    ]
)

pm10_importance = pd.DataFrame(
    xai_details["pm10"][
        "feature_importance"
    ].items(),
    columns=[
        "Feature",
        "Importance"
    ]
)


col1, col2 = st.columns(2)


with col1:

    st.write(
        "PM2.5 Model Feature Importance"
    )

    st.bar_chart(
        pm25_importance.set_index(
            "Feature"
        ),
        width="stretch"
    )


with col2:

    st.write(
        "PM10 Model Feature Importance"
    )

    st.bar_chart(
        pm10_importance.set_index(
            "Feature"
        ),
        width="stretch"
    )


# --------------------------------------------------
# Model Information
# --------------------------------------------------

st.subheader(
    "🤖 Model Information"
)

model_col1, model_col2 = st.columns(2)


with model_col1:

    st.metric(
        "PM2.5 Model",
        latest_prediction[
            "pm2_5_model"
        ]
    )


with model_col2:

    st.metric(
        "PM10 Model",
        latest_prediction[
            "pm10_model"
        ]
    )


# --------------------------------------------------
# Latest Environmental Conditions
# --------------------------------------------------

st.subheader(
    "🌡️ Latest Environmental Conditions"
)

environment_table = pd.DataFrame({

    "Parameter": [
        "Temperature",
        "Humidity",
        "NO₂",
        "CO",
        "O₃"
    ],

    "Value": [
        f"{latest_data.iloc[0]['temperature']:.2f}",
        f"{latest_data.iloc[0]['humidity']:.2f}",
        f"{latest_data.iloc[0]['no2']:.2f}",
        f"{latest_data.iloc[0]['co']:.2f}",
        f"{latest_data.iloc[0]['o3']:.2f}"
    ]

})


st.dataframe(
    environment_table,
    width="stretch",
    hide_index=True
)


# --------------------------------------------------
# Complete Prediction Information
# --------------------------------------------------

st.subheader(
    "📋 Complete Prediction Result"
)

result_table = pd.DataFrame({

    "Metric": [

        "PM2.5",
        "PM10",
        "AQI",
        "Category",
        "Confidence",
        "PM2.5 Dominant Feature",
        "PM2.5 Feature Importance",
        "PM10 Dominant Feature",
        "PM10 Feature Importance",
        "PM2.5 Model",
        "PM10 Model"

    ],

    "Value": [

        latest_prediction["pm2_5"],

        latest_prediction["pm10"],

        latest_prediction["aqi"],

        latest_prediction["category"],

        f"{latest_prediction['confidence'] * 100:.2f}%",

        latest_prediction[
            "pm2_5_dominant_feature"
        ],

        f"{latest_prediction['pm2_5_feature_importance'] * 100:.2f}%",

        latest_prediction[
            "pm10_dominant_feature"
        ],

        f"{latest_prediction['pm10_feature_importance'] * 100:.2f}%",

        latest_prediction[
            "pm2_5_model"
        ],

        latest_prediction[
            "pm10_model"
        ]

    ]

})


st.dataframe(
    result_table,
    width="stretch",
    hide_index=True
)


# --------------------------------------------------
# System Status
# --------------------------------------------------

st.divider()

st.success(
    "✓ AirTwinNet prediction pipeline is operational."
)

st.caption(
    "Data → Feature Engineering → ML Model → "
    "Prediction → AQI → XAI → Historical Monitoring "
    "→ Temporal Forecasting → Decision Support"
)