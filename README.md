# Real_time_fraud_detection

Real-Time Fraud Detection

Simulating incoming credit card transactions and predicting fraud status with a trained machine learning model.

---

Features

- Machine Learning Model: Random Forest Classifier trained on imbalanced credit card data  
- SMOTE Oversampling: Balances the dataset for better fraud detection recall  
- Live Transaction Stream: Processes one transaction at a time in real-time  
- Fraud Alerts: Immediate feedback for each transaction (fraud or not)  
- Risk Score & Fraud Counter: Displays confidence score and fraud count dynamically  
- Deployed via Streamlit Cloud: Fully functional web interface accessible publicly

---
StreamlitApp : " https://realtimefrauddetection-fgmefjss44odbzews9tw4f.streamlit.app/ "

Model Performance (Compared)

| Model               | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|-----------|--------|----------|---------|
|   Random Forest     | 0.42      | 0.87   | 0.57     | 0.9739 |
| XGBoost             | 0.24      | 0.89   | 0.38     | 0.9781 |
| Logistic Regression | 0.06      | 0.92   | 0.11     | 0.9698 |

> Random Forest was selected for deployment based on best F1-score and balanced fraud detection performance.
>
> Future Work (Optional)

- Deploy also on Hugging Face Spaces
- Add email alert

