import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_input, clean_column_names

# Load model and features
model = joblib.load("xgboost_readmission_model.pkl")
feature_names = joblib.load("model_features.pkl")

# Title
st.title("🏥 30-Day Readmission Predictor")
st.markdown("Upload patient data below (CSV) to predict risk of hospital readmission within 30 days.")

# ✅ File uploader (must come before the if-block)
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully.")

        # Preprocessing
        input_df = preprocess_input(raw_df.copy())
        input_df.columns = clean_column_names(input_df.columns)

        # Align features
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Predict
        probs = model.predict_proba(input_df)[:, 1]
        preds = (probs >= 0.3).astype(int)

        # Results
        raw_df["Readmission_Risk_Prob"] = probs.round(3)
        raw_df["Readmission_Predicted"] = ["Yes" if p == 1 else "No" for p in preds]

        st.subheader("📋 Prediction Results")
        st.dataframe(raw_df)

        csv = raw_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", csv, "readmission_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")
