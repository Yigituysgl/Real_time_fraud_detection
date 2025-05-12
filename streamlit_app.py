import streamlit as st
import pandas as pd
import joblib
import time


model = joblib.load("fraud_rf_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("test_data.csv")  

df = load_data()
current_index = st.session_state.get('index', 0)


st.title("💳 Real-Time Fraud Detection")
st.write("Simulating incoming transactions and predicting fraud status.")

if st.button("Next Transaction"):
    if current_index < len(df):
        row = df.iloc[current_index]
        X = row.drop('Class')  # Drop label
        y_true = row['Class']
        y_pred = model.predict([X])[0]
        st.write(f"**Transaction #{current_index + 1}:**")
        st.dataframe(pd.DataFrame([X]))
        st.success("✅ Not Fraud") if y_pred == 0 else st.error("🚨 Fraud Detected!")
        st.caption(f"Actual Label: {int(y_true)}")
        st.session_state.index = current_index + 1
    else:
        st.info("All transactions simulated.")

