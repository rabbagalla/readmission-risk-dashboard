# 🏥 30-Day Hospital Readmission Predictor

This project is a real-time predictive dashboard built with **Streamlit**, using a trained **XGBoost** model to identify patients at risk of hospital readmission within 30 days. The model was trained on the publicly available **Diabetes 130-US hospitals dataset** and deployed for clinical decision support.

## 🔍 Features

- Upload CSV file of patient-level hospital data
- Automated preprocessing (diagnosis mapping, medication encoding, etc.)
- Risk prediction with probability scores
- Threshold-based classification (`Yes` if probability ≥ 0.30)
- Downloadable results
- Real-time in the cloud

## 🚀 Live App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-STREAMLIT-APP-LINK-HERE)

## 📁 File Structure

📦 my_readmission_app/
├── app.py
├── preprocessing.py
├── model_features.pkl
├── xgboost_readmission_model.pkl
├── requirements.txt
└── README.md

## 📥 Sample Input Format

Upload a CSV file with columns like:

```csv
age,gender,race,diag_1,diag_2,diag_3,time_in_hospital,num_lab_procedures,...

🤝 Acknowledgements
Dataset: UCI Diabetes 130-US hospitals
