import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

import xgboost as xgb

# Load Booster model and wrap it in XGBClassifier
booster = xgb.Booster()
booster.load_model("xgb_booster_model.json")

model = xgb.XGBClassifier()
model._Booster = booster
model._le = None  # avoid label encoder issues

st.set_page_config(page_title="Readmission Risk Predictor", layout="wide")
st.title("🏥 Readmission Risk Dashboard")
st.markdown("Upload patient data to predict 30-day readmission risk.")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Data")
    st.dataframe(data.head())

    preds = model.predict(data)
    probs = model.predict_proba(data)[:, 1]

    data['Readmission_Risk'] = preds
    data['Risk_Probability'] = probs

    st.subheader("📊 Predictions")
    st.dataframe(data[['Readmission_Risk', 'Risk_Probability']])

    st.download_button("📥 Download Results", data.to_csv(index=False), file_name="predictions.csv")

    st.subheader("🔍 SHAP Explainability (First Patient)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    shap.initjs()
    st.pyplot(shap.force_plot(explainer.expected_value, shap_values[0], data.iloc[0], matplotlib=True))
