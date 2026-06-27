import os
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent


def resolve_artifact_path(filename):
    candidate_dirs = [
        BASE_DIR,
        BASE_DIR / "models",
        BASE_DIR.parent,
        BASE_DIR.parent / "models",
        Path.cwd(),
        Path.cwd() / "models",
    ]

    for directory in candidate_dirs:
        candidate = directory / filename
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find {filename} in any expected location")


# --- Load model pipeline and expected feature order (cached across reruns) ---
@st.cache_resource
def load_artifacts():
    # Pipeline: ColumnTransformer(OneHotEncoder on FuelType) -> LinearRegression
    model_path = resolve_artifact_path("used_car_price_prediction_model_v2.joblib")
    features_path = resolve_artifact_path("used_car_price_prediction_features_v2.joblib")

    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

model, FEATURES = load_artifacts()

# FuelType categories known to the fitted OneHotEncoder
FUEL_TYPES = ["CNG", "Diesel", "Petrol"]

# Initialise session state
if "input_features" not in st.session_state:
    st.session_state["input_features"] = {}


def app_sidebar():
    st.sidebar.header("Vehicle Details")

    age = st.sidebar.number_input("Age (months)", min_value=0, value=23, step=1)
    km = st.sidebar.number_input("Kilometres Driven", min_value=0, value=46986, step=500)
    fuel = st.sidebar.selectbox("Fuel Type", FUEL_TYPES, index=1)
    hp = st.sidebar.number_input("Horsepower (HP)", min_value=0, value=90, step=5)
    met_color = st.sidebar.selectbox("Metallic Colour", [0, 1], index=1,
                                     format_func=lambda x: "Yes" if x == 1 else "No")
    automatic = st.sidebar.selectbox("Automatic", [0, 1], index=0,
                                     format_func=lambda x: "Yes" if x == 1 else "No")
    cc = st.sidebar.number_input("Engine Size (CC)", min_value=0, value=2000, step=100)
    doors = st.sidebar.number_input("Doors", min_value=2, max_value=6, value=3, step=1)
    weight = st.sidebar.number_input("Weight (kg)", min_value=0, value=1165, step=10)

    def get_input_features():
        # Keys must match the trained feature names exactly
        return {
            "Age": int(age),
            "KM": int(km),
            "FuelType": fuel,
            "HP": int(hp),
            "MetColor": int(met_color),
            "Automatic": int(automatic),
            "CC": int(cc),
            "Doors": int(doors),
            "Weight": int(weight),
        }

    sdb_col1, sdb_col2 = st.sidebar.columns(2)
    with sdb_col1:
        predict_button = st.sidebar.button("Predict", key="predict")
    with sdb_col2:
        reset_button = st.sidebar.button("Reset", key="clear")

    if predict_button:
        st.session_state["input_features"] = get_input_features()
    if reset_button:
        st.session_state["input_features"] = {}
    return None


def app_body():
    title = ('<p style="font-family:arial, sans-serif; color:Black; font-size: 40px;">'
             '<b>Used Car Price Prediction</b></p>')
    st.markdown(title, unsafe_allow_html=True)

    if st.session_state["input_features"]:
        # Build a single-row DataFrame in the exact feature order the model expects
        row = pd.DataFrame([st.session_state["input_features"]])[FEATURES]
        price = float(model.predict(row)[0])
        st.success(f"**Estimated price:** {price:,.0f}")
        with st.expander("Inputs used"):
            st.dataframe(row.T.rename(columns={0: "value"}))
    else:
        st.info("Enter vehicle details in the sidebar and click **Predict**.")
    return None


def main():
    app_sidebar()
    app_body()
    return None


if __name__ == "__main__":
    main()