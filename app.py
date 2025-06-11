import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_input, clean_column_names

# Load model
model = joblib.load("xgboost_readmission_model.pkl")
feature_names = joblib.load("model_features.pkl")  # ← Keep this outside


if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully.")

        # Preprocess
        input_df = preprocess_input(raw_df.copy())
        input_df.columns = clean_column_names(input_df.columns)

        # Reindex AFTER preprocessing
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Predict
        probs = model.predict_proba(input_df)[:, 1]
        preds = (probs >= 0.3).astype(int)

        # Results
        output_df = raw_df.copy()
        output_df["Readmission_Risk_Prob"] = probs.round(3)
        output_df["Readmission_Predicted"] = ["Yes" if p == 1 else "No" for p in preds]

        st.subheader("📋 Prediction Results")
        st.dataframe(output_df[["Readmission_Risk_Prob", "Readmission_Predicted"]].join(raw_df))

        csv = output_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", csv, "readmission_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")

