import os
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Title
st.title("Milk Quality Prediction App")

# Description
st.write("🔬 This app predicts the **Remaining Lifespan** of milk using pH and CO₂ (ppm) values.")

# Upload CSV (optional)
uploaded_file = st.file_uploader("Upload your milk dataset (CSV)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.write(data.head())

    if 'pH' in data.columns and 'CO_ppm' in data.columns and 'Remaining_Lifespan' in data.columns:
        # Train model
        X = data[['pH', 'CO_ppm']]
        y = data['Remaining_Lifespan']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Evaluate model
        y_pred = model.predict(X)
        st.subheader("Model Evaluation")
        st.write(f"R²: {r2_score(y, y_pred):.2f}")
        st.write(f"MSE: {mean_squared_error(y, y_pred):.2f}")
        st.write(f"MAE: {mean_absolute_error(y, y_pred):.2f}")

        # Input for prediction
        st.subheader("Predict Lifespan")
        input_ph = st.number_input("Enter pH", min_value=0.0, max_value=14.0, value=6.5)
        input_co = st.number_input("Enter CO₂ (ppm)", min_value=0.0, value=1500.0)

        if st.button("Predict"):
            input_data = np.array([[input_ph, input_co]])
            prediction = model.predict(input_data)[0]
            st.success(f"Predicted Remaining Lifespan: {prediction:.2f} hours")
    else:
        st.error("CSV must include 'pH', 'CO_ppm', and 'Remaining_Lifespan' columns.")
else:
    st.info("⬆️ Upload a CSV file to begin.")
