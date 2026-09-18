import streamlit as st
import pandas as pd

from src.airtwinnet.components.prediction_service import (
    AirQualityPredictionService
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
# Load Prediction Service
# --------------------------------------------------

@st.cache_resource
def load_prediction_service():

    return AirQualityPredictionService(
        "artifacts/air_quality_model.pkl"
    )


# --------------------------------------------------
# Load Data
# --------------------------------------------------

@st.cache_data
def load_data():

    return pd.read_csv(
        "data/raw/air_quality_sample.csv"
    )


service = load_prediction_service()
data = load_data()

latest_data = data.tail(1)

result = service.predict(
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
# Current Location
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
        f"{result['pm2_5']:.2f}"
    )

with col2:

    st.metric(
        "PM10",
        f"{result['pm10']:.2f}"
    )

with col3:

    st.metric(
        "AQI",
        f"{result['aqi']:.2f}"
    )

with col4:

    st.metric(
        "Confidence",
        f"{result['confidence'] * 100:.2f}%"
    )


st.divider()


# --------------------------------------------------
# AQI Status
# --------------------------------------------------

st.subheader("Air Quality Status")

st.info(
    f"AQI Category: {result['category']}"
)


# --------------------------------------------------
# Explainable AI
# --------------------------------------------------

st.subheader("🧠 Explainable AI")

st.write(
    result["explanation"]
)


col1, col2 = st.columns(2)

with col1:

    st.write("**PM2.5 Dominant Feature**")

    st.write(
        f"{result['pm2_5_dominant_feature']} "
        f"({result['pm2_5_feature_importance'] * 100:.2f}%)"
    )

with col2:

    st.write("**PM10 Dominant Feature**")

    st.write(
        f"{result['pm10_dominant_feature']} "
        f"({result['pm10_feature_importance'] * 100:.2f}%)"
    )


# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

st.subheader("Feature Importance")

xai_details = service.xai.explain_model(
    service.model
)

pm25_importance = pd.DataFrame(
    xai_details["pm2_5"]["feature_importance"].items(),
    columns=["Feature", "Importance"]
)

pm10_importance = pd.DataFrame(
    xai_details["pm10"]["feature_importance"].items(),
    columns=["Feature", "Importance"]
)


col1, col2 = st.columns(2)

with col1:

    st.write("PM2.5 Model")

    st.bar_chart(
        pm25_importance.set_index("Feature")
    )

with col2:

    st.write("PM10 Model")

    st.bar_chart(
        pm10_importance.set_index("Feature")
    )


# --------------------------------------------------
# Current Input Data
# --------------------------------------------------

st.subheader("Latest Environmental Conditions")

display_columns = [
    "temperature",
    "humidity",
    "no2",
    "co",
    "o3"
]

st.dataframe(
    latest_data[display_columns],
    use_container_width=True
)


# --------------------------------------------------
# System Status
# --------------------------------------------------

st.divider()

st.success(
    "AirTwinNet prediction pipeline is operational."
)

st.caption(
    "Model → Prediction → AQI → XAI → Decision Support"
)