import streamlit as st
import pandas as pd
import joblib
import time


if "fraud_count" not in st.session_state:
    st.session_state.fraud_count = 0
if "normal_count" not in st.session_state:
    st.session_state.normal_count = 0
model = joblib.load("fraud_rf_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("test_data.csv")  

df = load_data()
current_index = st.session_state.get('index', 0)


st.title("💳 Real-Time Fraud Detection")
st.write("Simulating incoming transactions and predicting fraud status.")
st.subheader("📊 Live Summary")
col1, col2 = st.columns(2)
col1.metric("🚨 Fraud Detected", st.session_state.fraud_count)
col2.metric("✅ Normal Transactions", st.session_state.normal_count)


if st.button("Next Transaction"):
    if current_index < len(df):
        row = df.iloc[current_index]
        X = row.drop('Class')
        y_true = row['Class']

        
        with st.spinner("Processing transaction..."):
            time.sleep(1)  

        risk_score = model.predict_proba([X])[0][1]
            

        
        y_pred = model.predict([X])[0]

        
        st.write(f"### Transaction #{current_index + 1}")
        st.dataframe(pd.DataFrame([X]))

        if y_pred == 1:
            st.error("🚨 **FRAUD DETECTED!**")
            st.session_state.fraud_count += 1
        else:
            st.success("✅ Not Fraud")
            st.session_state.normal_count += 1

        st.caption(f"Actual Label: {int(y_true)}")
        st.caption(f"Model confidence (fraud risk): {risk_score:.2%}")

        
        st.session_state.index = current_index + 1
        progress = (current_index + 1) / len(df)
        st.progress(progress)

    else:
        st.info("✅ All transactions have been processed.")


