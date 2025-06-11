import streamlit as st
import pandas as pd
import joblib
import shap
import re
import streamlit.components.v1 as components
from preprocessing import preprocess_input, clean_column_names

# Load model and expected features
model = joblib.load("xgboost_readmission_model.pkl")
feature_names = joblib.load("model_features.pkl")

# UI Header
st.title("🏥 30-Day Hospital Readmission Predictor")
st.markdown("Upload a CSV file to predict the 30-day readmission risk for each patient.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload Patient CSV File", type="csv")

if uploaded_file is not None:
    try:
        # Load CSV
        raw_df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully.")

        # Ensure required columns exist
        required_cols = ['diag_1', 'diag_2', 'diag_3']
        missing = [col for col in required_cols if col not in raw_df.columns]
        if missing:
            st.error(f"❌ Missing required column(s): {', '.join(missing)}")
            st.stop()

        # Preprocess input
        input_df = preprocess_input(raw_df.copy())
        input_df.columns = clean_column_names(input_df.columns)
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Predict
        probs = model.predict_proba(input_df)[:, 1]
        preds = (probs >= 0.3).astype(int)

        # Combine with original data
        output_df = raw_df.copy()
        output_df["Readmission_Risk_Prob"] = probs.round(3)
        output_df["Readmission_Predicted"] = ["Yes" if p == 1 else "No" for p in preds]

        # Display predictions
        st.subheader("📋 Prediction Results")
        st.markdown("🔍 **Note:** A patient is predicted as `Yes` for readmission if probability ≥ 0.30.")
        result_cols = ["Readmission_Predicted", "Readmission_Risk_Prob"] + raw_df.columns.tolist()
        st.dataframe(output_df[result_cols])

        # Download button
        csv = output_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", csv, "readmission_predictions.csv", "text/csv")

        # SHAP Explainability for first patient
        st.subheader("🔎 SHAP Explanation: Patient 0")
        st.markdown("Below is the feature contribution breakdown for the first patient.")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        shap.initjs()
        force_plot = shap.plots.force(
            explainer.expected_value, shap_values[0], input_df.iloc[0], matplotlib=False
        )

        components.html(shap.getjs() + force_plot.html(), height=300)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
