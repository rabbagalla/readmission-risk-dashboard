import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_input, clean_column_names

# Load model and expected features
model = joblib.load("xgboost_readmission_model.pkl")
feature_names = joblib.load("model_features.pkl")

# Title & Instructions
st.title("🏥 30-Day Readmission Predictor")
st.markdown("Upload patient data below (CSV) to predict risk of hospital readmission within 30 days.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully.")

        # ✅ Check for required columns
        required_cols = ['diag_1', 'diag_2', 'diag_3']
        missing_cols = [col for col in required_cols if col not in raw_df.columns]
        if missing_cols:
            st.error(f"❌ Missing required column(s): {', '.join(missing_cols)}")
            st.stop()

        # Preprocess input
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

        # Display results
        st.subheader("📋 Prediction Results")
        st.markdown("🔍 **Note:** Patients predicted as `Yes` for readmission have a probability ≥ 0.30.")
        cols = ["Readmission_Predicted", "Readmission_Risk_Prob"] + raw_df.columns.tolist()
        st.dataframe(output_df[cols])

        # Download option
        csv = output_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", csv, "readmission_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
