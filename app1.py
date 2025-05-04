import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Title of the app
st.title("Milk Quality Prediction App")

# Description of the app
st.write("🔬 This app predicts the **Remaining Lifespan** and **Status** of milk using pH and CO₂ (ppm) values for raw and boiled milk.")

# Upload CSV files for raw and boiled milk datasets
uploaded_raw = st.file_uploader("Upload Raw Milk Dataset (CSV)", type="csv", key="raw")
uploaded_boiled = st.file_uploader("Upload Boiled Milk Dataset (CSV)", type="csv", key="boiled")

if uploaded_raw and uploaded_boiled:
    # Load datasets
    raw_df = pd.read_csv(uploaded_raw)
    boiled_df = pd.read_csv(uploaded_boiled)

    # Clean column names
    for df in [raw_df, boiled_df]:
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        df.dropna(inplace=True)

    # Verify required columns exist
    required_cols = ['Status', 'Remaining_lifespan', 'pH', 'CO_ppm']
    for col in required_cols:
        if col not in raw_df.columns or col not in boiled_df.columns:
            st.error(f"Required column '{col}' not found in one of the datasets.")
            st.stop()

    # Function to train and save models
    def train_and_save_models(df, milk_type, features, lifespan_target, status_target):
        X = df[features]
        y_lifespan = df[lifespan_target]
        y_status = df[status_target]

        # Split before scaling to prevent data leakage
        X_train, X_test, y_life_train, y_life_test = train_test_split(X, y_lifespan, test_size=0.2, random_state=42)
        y_status_train = y_status.loc[X_train.index]
        y_status_test = y_status.loc[X_test.index]

        # Scale using training data
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train regression model
        rf_regressor = RandomForestRegressor(random_state=42)
        rf_regressor.fit(X_train_scaled, y_life_train)

        # Train classification model
        rf_classifier = RandomForestClassifier(random_state=42)
        rf_classifier.fit(X_train_scaled, y_status_train)

        # Save models and scaler
        joblib.dump(scaler, f'scaler_{milk_type}.pkl')
        joblib.dump(rf_regressor, f'rf_regressor_{milk_type}.pkl')
        joblib.dump(rf_classifier, f'rf_classifier_{milk_type}.pkl')

        # Evaluate classifier
        y_pred_status = rf_classifier.predict(X_test_scaled)
        st.subheader(f"📊 Classification Report for {milk_type.capitalize()} Milk Status:")
        st.text(classification_report(y_status_test, y_pred_status))

    # Define features and targets
    features = ['pH', 'CO_ppm']
    lifespan_target = 'Remaining_lifespan'
    status_target = 'Status'

    # Train and save models
    train_and_save_models(raw_df, 'raw', features, lifespan_target, status_target)
    train_and_save_models(boiled_df, 'boiled', features, lifespan_target, status_target)

    # Prediction function
    def predict_shelf_life(milk_type, pH, CO_ppm):
        try:
            # Load models and scaler
            scaler = joblib.load(f'scaler_{milk_type}.pkl')
            rf_regressor = joblib.load(f'rf_regressor_{milk_type}.pkl')
            rf_classifier = joblib.load(f'rf_classifier_{milk_type}.pkl')
        except FileNotFoundError:
            st.error(f"Models for {milk_type} milk not found. Ensure models are trained and saved.")
            return None, None

        # Prepare input data and scale it
        input_data = np.array([[pH, CO_ppm]])
        input_data_scaled = scaler.transform(input_data)

        # Predict lifespan and status
        predicted_lifespan = rf_regressor.predict(input_data_scaled)[0]
        predicted_status = rf_classifier.predict(input_data_scaled)[0]

        return predicted_lifespan, predicted_status

    # Input for prediction
    milk_type = st.selectbox("Select Milk Type", ["raw", "boiled"])
    input_ph = st.number_input("Enter pH value", min_value=0.0, max_value=14.0, value=6.5)
    input_co = st.number_input("Enter CO₂ (ppm) value", min_value=0.0, value=1500.0)

    if st.button("Predict Shelf Life"):
        predicted_lifespan, predicted_status = predict_shelf_life(milk_type, input_ph, input_co)
        if predicted_lifespan and predicted_status:
            st.success(f"🕒 Predicted Remaining Shelf Life: {predicted_lifespan:.2f} hours")
            st.success(f"🥛 Milk Status: {predicted_status}")

else:
    st.info("⬆️ Upload Raw and Boiled Milk CSV files to begin.")
