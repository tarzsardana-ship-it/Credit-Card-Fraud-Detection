import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- LOAD MODEL ----------------
model = joblib.load("credit_card_fraud_model.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Banking Fraud Detection System")
st.caption("AI-powered real-time transaction monitoring")

st.divider()

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("System Status", "Active", "🟢")

with col2:
    st.metric("Model", "Pre-trained ML", "⚙️")

with col3:
    st.metric("Security Level", "High", "🔐")

st.divider()

tab1, tab2 = st.tabs(["Single Transaction", "CSV Upload"])

# ======================================================
# 🟢 SINGLE TRANSACTION
# ======================================================
with tab1:

    st.subheader("Analyze Single Transaction")

    time = st.number_input("Time", value=0.0)

    st.markdown("### V1 - V28 Features")

    v_features = []
    cols = st.columns(4)

    for i in range(1, 29):
        with cols[(i - 1) % 4]:
            v_features.append(
                st.number_input(f"V{i}", value=0.0, key=f"v{i}")
            )

    amount = st.number_input("Amount", value=0.0)

    if st.button("🔍 Analyze Transaction"):

        input_data = np.array([[time] + v_features + [amount]])

        try:
            pred = model.predict(input_data)[0]

            prob = model.predict_proba(input_data)[0][1]

            st.divider()
            st.subheader("📊 Risk Analysis")

            # Risk meter
            st.progress(min(int(prob * 100), 100))

            st.write(f"🔢 Fraud Probability: **{prob:.2f}**")

            # Business decision logic
            if prob > 0.7:
                action = "🚨 BLOCK TRANSACTION"
                st.error(action)
            elif prob > 0.4:
                action = "⚠️ SEND FOR REVIEW"
                st.warning(action)
            else:
                action = "✅ APPROVE TRANSACTION"
                st.success(action)

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ======================================================
# 🔵 CSV UPLOAD
# ======================================================
with tab2:

    st.subheader("Batch Fraud Detection (CSV)")

    file = st.file_uploader("Upload Transaction CSV", type=["csv"])

    if file is not None:

        try:
            df = pd.read_csv(file)

            st.write("📄 Data Preview")
            st.dataframe(df.head())

            expected_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

            missing = [c for c in expected_cols if c not in df.columns]

            if missing:
                st.error(f"Missing columns: {missing}")

            else:
                df = df[expected_cols]

                pred = model.predict(df)
                prob = model.predict_proba(df)[:, 1]

                df["Fraud"] = pred
                df["Risk Score"] = prob

                st.subheader("📊 Results Table")
                st.dataframe(df)

                fraud = (pred == 1).sum()
                normal = (pred == 0).sum()

                col1, col2 = st.columns(2)
                col1.metric("🚨 Fraud Cases", fraud)
                col2.metric("✅ Normal Cases", normal)

                st.bar_chart(
                    pd.DataFrame(
                        {"Count": [fraud, normal]},
                        index=["Fraud", "Normal"]
                    )
                )

        except Exception as e:
            st.error(f"CSV Error: {e}")