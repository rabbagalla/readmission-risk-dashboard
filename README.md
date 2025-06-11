# 🏥 Predicting 30-Day Hospital Readmissions with XGBoost & SHAP

This project builds a **real-time Streamlit dashboard** for predicting the risk of **30-day hospital readmission**, using the **Diabetes 130-US hospitals** dataset. It leverages **machine learning (XGBoost)** and **model explainability (SHAP)** to assist clinicians and population health managers in identifying high-risk patients.

> 🎯 Goal: Help reduce avoidable readmissions through proactive care targeting  
> 🧠 Model: XGBoost (tuned) with threshold-based classification  
> ☁️ Deployed via: Streamlit Cloud

---

## 📊 Project Overview

- 📁 **Dataset**: [UCI Diabetes 130-US Hospitals Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)  
- 🧹 **Preprocessing**: Custom diagnosis grouping, medication encoding, categorical handling  
- 🔍 **Modeling**: XGBoost with threshold tuning (0.3) for better minority class (readmission) recall  
- 📈 **Evaluation**: ROC-AUC, precision/recall, F1  
- 🔎 **Explainability**: SHAP for per-patient feature influence  
- 🧪 **UI**: Upload raw CSV → Predict risk → Download results

---

## 🚀 Live App Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-STREAMLIT-APP-LINK)

Try the hosted app — no installation needed. Upload a `.csv` file and get instant readmission risk scores.

---

## 🧠 Model Performance Summary

| Model               | Accuracy | Recall (Readmitted) | ROC-AUC |
|--------------------|----------|----------------------|---------|
| Logistic Regression | 67%     | **0.55**             | 0.67    |
| Random Forest       | 89%     | 0.02                 | 0.64    |
| XGBoost + Threshold 0.3 | 69% | **0.53**             | **0.66** ✅

✅ **XGBoost** with threshold tuning gave the best tradeoff between recall and precision for readmission class.

---

## 🧮 SHAP Explainability

Each prediction includes a SHAP force plot that shows:

- What features pushed the readmission risk **higher**
- What features pulled it **lower**

Example:

![SHAP Screenshot](./assets/shap_sample.png)

This adds transparency to the decision-making process, especially useful for clinicians or care coordinators.

---

## 🧬 Sample Columns Required in Input CSV

```csv
age,gender,race,diag_1,diag_2,diag_3,time_in_hospital,num_lab_procedures,...
