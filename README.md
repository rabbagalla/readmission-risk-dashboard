# 🏥 30-Day Hospital Readmission Predictor (XGBoost + SHAP + Streamlit)

This project is a **real-time Streamlit dashboard** that predicts the risk of **30-day hospital readmission** for patients, using the [Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008). It combines a tuned **XGBoost model**, deep feature preprocessing, and **SHAP explainability** — deployed to the cloud for real-world usability.

---

## 🔎 Project Highlights

| Feature                     | Details |
|----------------------------|---------|
| **Model**                  | XGBoost with threshold tuning (0.3) |
| **Explainability**         | SHAP force plots for per-patient reasoning |
| **Deployment**             | Streamlit Cloud |
| **Data**                   | UCI Diabetes 130-US Hospitals Dataset |
| **Use Case**               | Predicting 30-day hospital readmission risk |

---

## 🚀 Live Demo

👉 [**Click here to try the live app**](https://your-streamlit-app-link)

> Upload a CSV with patient data and get instant readmission predictions with SHAP explanations.

---

## 📊 Model Performance Summary

| Model                | Accuracy | Recall (Readmitted) | ROC-AUC |
|---------------------|----------|----------------------|---------|
| Logistic Regression | 67%      | **0.55**             | 0.67    |
| Random Forest        | 89%      | 0.02                 | 0.64    |
| **XGBoost (0.3 thresh)** | 69% | **0.53**             | **0.66** ✅

✅ XGBoost achieved the best **recall on minority class** (readmitted patients) — critical in healthcare.

---

## 📦 Project Structure

📦 my_readmission_app/
├── app.py # Streamlit dashboard
├── preprocessing.py # Data cleaning + feature engineering
├── model_features.pkl # List of features used during training
├── xgboost_readmission_model.pkl # Trained model file
├── test_input.csv # Sample input file
├── requirements.txt
└── README.md


---

## 📥 Input Format

Upload a `.csv` with patient records like:

```csv
age,gender,race,diag_1,diag_2,diag_3,time_in_hospital,num_lab_procedures,...

👉 Try with test_input.csv
🧠 SHAP Explainability
Each prediction is explained using SHAP force plots, showing which features pushed the patient toward or away from readmission risk.

Example:

🔍 Use the dropdown to select any patient and explore the reasoning behind their prediction.


🧰 Tech Stack
Python, Pandas, NumPy

XGBoost, Scikit-learn

SHAP for explainability

Streamlit for interactive UI

Joblib for model persistence

✅ What I Learned
🏥 Handling imbalanced clinical data using threshold tuning

🧠 SHAP for interpretable machine learning in healthcare

🛠 Building end-to-end apps from raw data to dashboard

🚀 Real-world deployment with Streamlit Cloud

🙋‍♂️ About Me
I'm a Master’s graduate in Health Data Science, passionate about making machine learning actionable and explainable in healthcare.🛠 Future Improvements
SHAP summary plot for all patients

Customizable risk thresholds

Add patient name/ID selection

Integrated CSV formatting guide in-app

🙏 Acknowledgements
Dataset: UCI Machine Learning Repository

SHAP: Lundberg & Lee (NIPS 2017)

Streamlit Community

yaml
Copy
Edit

