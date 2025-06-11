import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# Load Booster (raw model)
booster = xgb.Booster()
booster.load_model("xgb_booster_model.json")

st.set_page_config(page_title="Readmission Risk Predictor", layout="wide")
st.title("🏥 Readmission Risk Dashboard")
st.markdown("Upload patient data to predict 30-day readmission risk.")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Data")
    st.dataframe(data.head())

    # Convert to DMatrix (XGBoost's native format)
    dmatrix = xgb.DMatrix(data)

    # Predict readmission risk
    predictions = booster.predict(dmatrix)
    labels = (predictions > 0.5).astype(int)

    data['Readmission_Risk'] = labels
    data['Risk_Probability'] = predictions

    st.subheader("📊 Predictions")
    st.dataframe(data[['Readmission_Risk', 'Risk_Probability']])

    st.download_button("📥 Download Results", data.to_csv(index=False), file_name="predictions.csv")

    # SHAP - Optional (might crash on Streamlit Cloud if unsupported)
    try:
        st.subheader("🔍 SHAP Explanation (First Patient)")
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(data)
        shap.initjs()
        st.pyplot(shap.force_plot(explainer.expected_value, shap_values[0], data.iloc[0], matplotlib=True))
    except Exception as e:
        st.warning(f"SHAP plot not supported in this environment: {e}")
