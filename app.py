import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- LOAD MODEL ----------------
model = joblib.load("credit_card_fraud_model.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bank Fraud Detection System",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Banking Fraud Detection System")
st.caption("AI-powered real-time fraud monitoring dashboard")

st.divider()

# ---------------- TOP METRICS ----------------
col1, col2, col3 = st.columns(3)

col1.metric("System Status", "ACTIVE", "🟢")
col2.metric("Model", "Pre-trained ML", "⚙️")
col3.metric("Security Level", "HIGH", "🔐")

st.divider()

tab1, tab2 = st.tabs(["Single Transaction", "Batch Analysis (CSV)"])

# ======================================================
# 🟢 SINGLE TRANSACTION (CLEAN UI - NO V FEATURES)
# ======================================================
with tab1:

    st.subheader("Analyze Transaction")

    time = st.number_input("Transaction Time", value=0.0)
    amount = st.number_input("Transaction Amount", value=0.0)

    if st.button("🔍 Analyze"):

        # V1–V28 hidden (filled with 0)
        v_features = [0.0] * 28

        input_data = np.array([[time] + v_features + [amount]])

        prob = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]

        st.divider()
        st.subheader("📊 Risk Analysis")

        # Risk meter
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

# ======================================================
# 🔵 CSV UPLOAD (SAFE + VISUALIZATION)
# ======================================================
with tab2:

    st.subheader("Batch Fraud Detection")

    file = st.file_uploader("Upload Transaction CSV", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)

        st.write("📄 Data Preview")
        st.dataframe(df.head())

        # Ensure correct structure
        if "Time" not in df.columns or "Amount" not in df.columns:
            st.error("CSV must contain 'Time' and 'Amount'")
        else:

            # fill missing V1–V28
            for i in range(1, 29):
                if f"V{i}" not in df.columns:
                    df[f"V{i}"] = 0.0

            # reorder columns
            cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
            df = df[cols]

            pred = model.predict(df)
            prob = model.predict_proba(df)[:, 1]

            df["Fraud Prediction"] = pred
            df["Risk Score"] = prob

            st.subheader("📊 Results")
            st.dataframe(df)

            fraud_count = (pred == 1).sum()
            normal_count = (pred == 0).sum()

            col1, col2 = st.columns(2)
            col1.metric("🚨 Fraud Cases", fraud_count)
            col2.metric("✅ Normal Cases", normal_count)

            st.divider()

            # Visualization
            chart_data = pd.DataFrame(
                {"Count": [fraud_count, normal_count]},
                index=["Fraud", "Normal"]
            )

            st.bar_chart(chart_data)