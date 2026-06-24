import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("credit_card_fraud_model.pkl")

st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("🏦 AI Fraud Detection System (Production Mode)")
st.caption("Real-time banking transaction monitoring")

st.divider()

# INPUT
time = st.number_input("Transaction Time", value=0.0)
amount = st.number_input("Transaction Amount", value=0.0)

# SMART INPUT BUILDER
def build_input(time, amount):
    v_features = [0.0] * 28  # neutral baseline
    return np.array([[time] + v_features + [amount]])

# PREDICTION
if st.button("Analyze Transaction"):

    input_data = build_input(time, amount)

    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    st.divider()
    st.subheader("📊 Risk Analysis")

    st.progress(min(int(prob * 100), 100))

    col1, col2 = st.columns(2)

    with col1:
        if pred == 1:
            st.error("🚨 FRAUD DETECTED")
        else:
            st.success("✅ SAFE TRANSACTION")

        st.write(f"Risk Score: **{prob:.2f}**")

    with col2:
        if prob > 0.7:
            action = "BLOCK TRANSACTION ❌"
        elif prob > 0.4:
            action = "SEND FOR REVIEW ⚠️"
        else:
            action = "APPROVE TRANSACTION ✅"

        st.info(f"💼 Action: {action}")