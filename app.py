import streamlit as st
import pandas as pd
import joblib

st.title("US Accident Severity Prediction")

try:
    model = joblib.load("best_model.pkl")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error("Error loading model:")
    st.exception(e)
    st.stop()

names = [
    "Start_Lat", "Start_Lng", "Distance(mi)", "Temperature(F)",
    "Wind_Chill(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)",
    "Wind_Speed(mph)", "Precipitation(in)", "Year", "Month", "Hour"
]

values = [st.number_input(name) for name in names]

if st.button("Predict"):
    data = pd.DataFrame([values], columns=names)

    try:
        prediction = model.predict(data)[0]
        st.success(f"Predicted Severity: {prediction}")
    except Exception as e:
        st.error("Error while making prediction:")
        st.exception(e)
