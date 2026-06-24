import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("credit_card_fraud_model.pkl")

st.title("💳 Credit Card Fraud Detection System")

tab1, tab2 = st.tabs(["Single Transaction", "CSV Upload"])

# ---------------- SINGLE INPUT ----------------
with tab1:
    st.subheader("Enter Transaction Details")

    time = st.number_input("Time")
    amount = st.number_input("Amount")

    if st.button("Predict Single"):
        input_data = np.array([[time, amount] + [0]*28])
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("Fraud Transaction 🚨")
        else:
            st.success("Normal Transaction ✅")

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

        st.write(f"Fraud: {fraud_count}")
        st.write(f"Normal: {normal_count}")