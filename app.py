import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("credit_card_fraud_model.pkl")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")

tab1, tab2 = st.tabs(["Single Transaction", "CSV Upload"])

# ---------------- SINGLE INPUT ----------------
with tab1:
    st.subheader("Enter Transaction Details")

    time = st.number_input("Time", value=0.0)

    st.markdown("### V1 - V28 Features")

    cols = st.columns(4)
    v_features = []

    for i in range(1, 29):
        with cols[(i - 1) % 4]:
            value = st.number_input(
                f"V{i}",
                value=0.0,
                format="%.6f",
                key=f"v{i}"
            )
            v_features.append(value)

    amount = st.number_input("Amount", value=0.0)

    if st.button("Predict Single"):

        input_data = np.array([[time] + v_features + [amount]])

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("🚨 Fraud Transaction Detected")
        else:
            st.success("✅ Normal Transaction")

# ---------------- CSV UPLOAD ----------------
with tab2:
    st.subheader("Upload CSV File")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)

        prediction = model.predict(df)

        df["Prediction"] = prediction

        st.write(df)

        fraud_count = (prediction == 1).sum()
        normal_count = (prediction == 0).sum()

        st.write(f"🚨 Fraud Transactions: {fraud_count}")
        st.write(f"✅ Normal Transactions: {normal_count}")

        chart_data = pd.DataFrame(
            {"Count": [fraud_count, normal_count]},
            index=["Fraud", "Normal"]
        )

        st.bar_chart(chart_data)