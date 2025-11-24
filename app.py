import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("delay_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="E-Commerce Delay Prediction", layout="centered")

st.title("🚚 E-Commerce Delivery Delay Prediction App")
st.write("Fill the inputs below to check if the delivery will be delayed.")

distance = st.number_input("Distance (km)", min_value=0.0, step=1.0)
weight = st.number_input("Package Weight (kg)", min_value=0.1, step=0.1)
cost = st.number_input("Delivery Cost", min_value=0.0, step=1.0)
rating = st.number_input("Delivery Rating (1–5)", min_value=1.0, max_value=5.0, step=0.1)
delay_hours = st.number_input("Delay Hours (delivery_time - expected_time)", step=0.1)

weather = st.selectbox("Weather Condition", 
    ["clear", "rainy", "foggy", "humid", "stormy", "other"])
partner = st.selectbox("Delivery Partner", 
    ["delhivery","xpressbees","shadowfax","ekart","dtdc","bluedart","other"])
vehicle = st.selectbox("Vehicle Type",
    ["bike","van","ev van","truck","scooter","other"])
mode = st.selectbox("Delivery Mode",
    ["same day","express","two day","standard"])
region = st.selectbox("Region",
    ["north","south","east","west","central","other"])

def preprocess():
    input_data = {
        "distance_km": distance,
        "package_weight_kg": weight,
        "delivery_cost": cost,
        "delivery_rating": rating,
        "delay_hours": delay_hours,
        "weather_condition_" + weather: 1,
        "delivery_partner_" + partner: 1,
        "vehicle_type_" + vehicle: 1,
        "delivery_mode_" + mode: 1,
        "region_" + region: 1
    }

    cols = list(model.feature_names_in_)
    row = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)

    # Fill values
    for col in row.columns:
        if col in input_data:
            row[col] = input_data[col]
        elif col in ["distance_km","package_weight_kg","delivery_cost","delivery_rating","delay_hours"]:
            row[col] = input_data[col]

    return row

if st.button("Predict Delay"):
    X = preprocess()
    pred = model.predict(X)[0]

    if pred == 1:
        st.error("🔴 DELIVERY WILL BE DELAYED")
    else:
        st.success("🟢 DELIVERY WILL BE ON TIME")
