import streamlit as st
import numpy as np
import pandas as pd
import joblib

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="California Housing Price Prediction",
    page_icon="🏡",
    layout="centered"
)

# --- SIMPLE CUSTOM STYLE ---
st.markdown(
    """
    <style>
    .main {
        padding-top: 2rem;
    }
    .stApp {
        background: radial-gradient(circle at top left, #1f2933, #020617);
    }
    .title-card {
        background: #111827;
        padding: 1.5rem 2rem;
        border-radius: 1rem;
        border: 1px solid #374151;
        box-shadow: 0 20px 40px rgba(0,0,0,0.35);
    }
    .title-card h1 {
        font-size: 2.1rem;
        margin-bottom: 0.3rem;
    }
    .title-card p {
        color: #9ca3af;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-card">
        <h1>🏡 California Housing Price Prediction</h1>
        <p>Enter the house characteristics to estimate the median house value.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")  # small spacer

# --- LOAD ARTIFACTS (scaler + model + columns) ---
artifacts = joblib.load("model.pkl")
scaler = artifacts["scaler"]
model = artifacts["model"]
columns = artifacts["columns"]

ocean_categories = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

# --- INPUTS ---
col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude", value=-118.0, step=0.1)
    latitude = st.number_input("Latitude", value=34.0, step=0.1)
    housing_median_age = st.number_input(
        "Housing median age", min_value=1, max_value=100, value=30, step=1, format="%d"
    )
    total_rooms = st.number_input(
        "Total rooms", min_value=1, max_value=50000, value=2000, step=10, format="%d"
    )

with col2:
    total_bedrooms = st.number_input(
        "Total bedrooms", min_value=1, max_value=10000, value=400, step=1, format="%d"
    )
    population = st.number_input(
        "Population", min_value=1, max_value=50000, value=800, step=10, format="%d"
    )
    households = st.number_input(
        "Households", min_value=1, max_value=10000, value=300, step=5, format="%d"
    )
    median_income = st.number_input(
        "Median income (10k USD)", min_value=0.0, max_value=20.0, value=3.0, step=0.1
    )

# categorical
ocean_prox = st.selectbox("Ocean proximity", ocean_categories)

# --- BUILD INPUT ROW IN SAME ORDER AS TRAINING ---
row = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "<1H OCEAN": 0,
    "INLAND": 0,
    "ISLAND": 0,
    "NEAR BAY": 0,
    "NEAR OCEAN": 0,
}
row[ocean_prox] = 1  # set chosen category to 1

X_raw = pd.DataFrame([row])[columns]  # ensure correct column order

# --- PREDICTION ---
if st.button("Predict price"):
    X_scaled = scaler.transform(X_raw.values)
    pred = model.predict(X_scaled)[0]
    st.success(f"Estimated median house value: **${pred:,.0f}**")
