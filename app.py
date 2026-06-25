import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

model = joblib.load("credit_card_fraud_model.pkl")

# ---------------- HEADER ----------------

st.title("🛡️ Credit Card Fraud Detection System")
st.markdown("### AI Powered Transaction Risk Analysis")

st.divider()

# ---------------- METRICS ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🟢 System Status: ACTIVE")

with col2:
    st.info("🤖 Detection Engine: ML Model")

with col3:
    st.info("🔐 Security Level: HIGH")

st.divider()

# ---------------- INPUTS ----------------

left, right = st.columns(2)

with left:

    st.subheader("Transaction Details")

    time = st.number_input(
        "Transaction Time",
        min_value=0.0,
        value=0.0
    )

with right:

    st.subheader("Payment Details")

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0
    )

st.write("")

# ---------------- PREDICTION ----------------

if st.button("🔍 Analyze Transaction"):

    v_features = [0.0] * 28

    input_data = np.array(
        [[time] + v_features + [amount]]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    st.subheader("📊 Risk Analysis")

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.pie(
        [probability, 1 - probability],
        labels=["Fraud Risk", "Safe"],
        autopct="%1.1f%%"
    )

    st.pyplot(fig)

    st.progress(int(probability * 100))

    st.metric(
        label="Fraud Probability",
        value=f"{probability:.2%}"
    )

    st.write("")

    if prediction == 1:

        st.error("🚨 FRAUD TRANSACTION DETECTED")

        if probability > 0.7:
            st.warning("Recommended Action: BLOCK TRANSACTION")

    else:

        st.success("✅ NORMAL TRANSACTION")

        st.success("Recommended Action: APPROVE TRANSACTION")

# ---------------- SIDEBAR ----------------

st.sidebar.title("Fraud Detection")

st.sidebar.write("Machine Learning Based Fraud Detection")

st.sidebar.success("System Ready")