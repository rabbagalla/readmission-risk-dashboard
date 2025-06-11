# app.py

import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_input, clean_column_names

# Load model
model = joblib.load("xgboost_readmission_model.pkl")

st.title("🏥 30-Day Readmission Predictor")
st.markdown("Upload patient data below (CSV) to predict risk of hospital readmission within 30 days.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully.")

        # Preprocess data
        input_df = preprocess_input(raw_df.copy())
        input_df.columns = clean_column_names(input_df.columns)

        # Align columns with model input
        model_features = model.get_booster().feature_names
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        # Predict
        probs = model.predict_proba(input_df)[:, 1]
        preds = (probs >= 0.3).astype(int)  # Custom threshold

        # Display results
        output_df = raw_df.copy()
        output_df["Readmission_Risk_Prob"] = probs.round(3)
        output_df["Readmission_Predicted"] = ["Yes" if p == 1 else "No" for p in preds]

        st.subheader("📋 Prediction Results")
        st.dataframe(output_df[["Readmission_Risk_Prob", "Readmission_Predicted"]].join(raw_df))

        # Download option
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv, "readmission_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")
