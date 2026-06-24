import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("credit_card_fraud_model.pkl", "rb"))

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# HEADER
st.title("🏦 Banking Fraud Detection Dashboard")
st.caption("AI-powered real-time transaction monitoring system")

st.divider()

# LAYOUT (Dashboard style)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("System Status", "Active", "🟢")

with col2:
    st.metric("Model Type", "Pre-trained ML", "⚙️")

with col3:
    st.metric("Risk Engine", "Live", "⚡")

st.divider()

# INPUT SECTION
st.subheader("💳 Transaction Details")

col1, col2 = st.columns(2)

with col1:
    time = st.number_input("Transaction Time (seconds)")
    amount = st.number_input("Transaction Amount")

with col2:
    v1 = st.number_input("V1")
    v2 = st.number_input("V2")
    v3 = st.number_input("V3")

# BUTTON
if st.button("🔍 Analyze Transaction", use_container_width=True):

    input_data = np.array([[time, amount, v1, v2, v3] + [0]*25])

    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    st.divider()
    st.subheader("📊 Risk Analysis Result")

    # RISK BAR STYLE
    st.progress(min(int(prob * 100), 100))

    col1, col2 = st.columns(2)

    with col1:
        if pred == 1:
            st.error("🚨 FRAUD DETECTED")
            st.write(f"Risk Score: {prob:.2f}")
        else:
            st.success("✅ SAFE TRANSACTION")
            st.write(f"Risk Score: {prob:.2f}")

    with col2:
        if prob > 0.7:
            action = "BLOCK TRANSACTION ❌"
        elif prob > 0.4:
            action = "SEND FOR REVIEW ⚠️"
        else:
            action = "APPROVE TRANSACTION ✅"

        st.info(f"💼 Recommended Action: {action}")

st.divider()

st.caption("⚠️ AI system for banking fraud detection - demo purpose only")