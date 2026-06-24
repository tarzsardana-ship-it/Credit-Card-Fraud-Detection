import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- LOAD MODEL ----------------
model = joblib.load("credit_card_fraud_model.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("🏦 AI Banking Fraud Detection Dashboard")
st.caption("Real-time transaction risk monitoring system")

st.divider()

# ---------------- TOP METRICS ----------------
col1, col2, col3 = st.columns(3)

col1.metric("System Status", "ACTIVE", "🟢")
col2.metric("Model", "ML Powered", "⚙️")
col3.metric("Security Level", "HIGH", "🔐")

st.divider()

time = st.number_input("Transaction Time", value=0.0)
amount = st.number_input("Transaction Amount", value=0.0)

if st.button("🔍 Analyze Transaction"):

    v_features = [0.0] * 28
    input_data = np.array([[time] + v_features + [amount]])

    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    st.divider()
    st.subheader("📊 Risk Analysis Dashboard")

    # ---------------- GAUGE STYLE ----------------
    fig, ax = plt.subplots()
    ax.pie(
        [prob, 1 - prob],
        labels=["Risk", "Safe"],
        autopct="%1.1f%%",
        colors=["red", "green"]
    )
    ax.set_title("Fraud Risk Distribution")

    st.pyplot(fig)

    # ---------------- RISK BAR ----------------
    st.progress(min(int(prob * 100), 100))
    st.write(f"🔢 Fraud Probability: **{prob:.2f}**")

    # ---------------- RESULT CARD ----------------
    col1, col2 = st.columns(2)

    with col1:
        if pred == 1:
            st.error("🚨 FRAUD TRANSACTION DETECTED")
        else:
            st.success("✅ NORMAL TRANSACTION")

    with col2:
        if prob > 0.7:
            st.warning("💼 ACTION: BLOCK TRANSACTION ❌")
        elif prob > 0.4:
            st.info("💼 ACTION: SEND FOR REVIEW ⚠️")
        else:
            st.success("💼 ACTION: APPROVE TRANSACTION ✅")

# ---------------- FOOTER ----------------
st.divider()
st.caption("⚠️ AI-based fraud detection system for demonstration purposes")