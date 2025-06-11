import streamlit as st
import pandas as pd
import joblib
import shap
import re
from preprocessing import preprocess_input, clean_column_names

# Load model and expected features
model = joblib.load("xgboost_readmission_model.pkl")
feature_names = joblib.load("model_features.pkl")

# Streamlit UI
st.title("🏥 30-Day Hospital Readmission Predictor")
st.markdown("Upload patient data (CSV) to predict risk of hospital readmission within 30 days.")

# File upload
uploaded_file = st.file_uploader("📤 Upload CSV", type="csv")

if uploaded_file is not None:
    try:
        # Read file
        raw_df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully.")

        # Check for required columns
        required_cols = ['diag_1', 'diag_2', 'diag_3']
        missing_cols = [col for col in required_cols if col not in raw_df.columns]
        if missing_cols:
            st.error(f"❌ Missing required column(s): {', '.join(missing_cols)}")
            st.stop()

        # Preprocessing
        input_df = preprocess_input(raw_df.copy())
        input_df.columns = clean_column_names(input_df.columns)
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Predict
        probs = model.predict_proba(input_df)[:, 1]
        preds = (probs >= 0.3).astype(int)

        # Format output
        output_df = raw_df.copy()
        output_df["Readmission_Risk_Prob"] = probs.round(3)
        output_df["Readmission_Predicted"] = ["Yes" if p == 1 else "No" for p in preds]

        # Display
        st.subheader("📋 Prediction Results")
        st.markdown("🔍 **Note:** Patients predicted as `Yes` have probability ≥ 0.30.")
        display_cols = ["Readmission_Predicted", "Readmission_Risk_Prob"] + raw_df.columns.tolist()
        st.dataframe(output_df[display_cols])

        # Download button
        csv = output_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", csv, "readmission_predictions.csv", "text/csv")

        # SHAP Explanation
        st.subheader("🔎 SHAP Explanation: Patient 0")
        st.markdown("Feature contributions to predicted readmission risk.")
        input_df.columns = clean_column_names(input_df.columns)  # Ensure clean names
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        shap.initjs()
        shap_html = shap.plots._force.AdditiveForceVisualizer(
            explainer.expected_value, shap_values[0], input_df.iloc[0]
        ).html()

        st.components.v1.html(shap_html, height=300)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
