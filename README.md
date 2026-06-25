# 🏦 Credit Card Fraud Detection System

An AI-powered Credit Card Fraud Detection System built using Machine Learning and deployed with Streamlit. The project analyzes transaction data and predicts whether a credit card transaction is legitimate or fraudulent, helping financial institutions identify suspicious activities in real time.

## 🚀 Live Demo

https://r4hxductdf2au4eupvczco.streamlit.app/

## 📂 GitHub Repository

https://github.com/tarzsardana-ship-it/Credit-Card-Fraud-Detection

---

## 📌 Project Overview
This project applies Machine Learning techniques to identify fraudulent transactions and distinguish them from legitimate ones. Since the dataset is highly imbalanced, Random Undersampling was used to create a balanced dataset before model training.
A Logistic Regression classifier was trained and deployed through an interactive Streamlit dashboard for real-time fraud analysis.

---

## ✨ Key Features

✔ Credit Card Fraud Detection using Machine Learning

✔ Interactive Streamlit Web Application

✔ Real-Time Transaction Risk Analysis

✔ Fraud Probability Prediction

✔ Visual Risk Distribution Dashboard

✔ Automated Transaction Recommendation System

✔ Handling Imbalanced Dataset using Undersampling

✔ Model Deployment with Streamlit Cloud

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* Scikit-Learn
* Joblib
* Matplotlib
* Streamlit

### Machine Learning Algorithm

* Logistic Regression

---

## 📊 Dataset Information

The project uses the Credit Card Fraud Detection Dataset containing anonymized credit card transaction records.

### Dataset Features

* Time
* V1 – V28 (PCA-transformed features)
* Amount
* Class (Target Variable)

### Target Classes

| Class | Meaning                |
| ----- | ---------------------- |
| 0     | Legitimate Transaction |
| 1     | Fraudulent Transaction |

The dataset is highly imbalanced, with fraudulent transactions representing only a small percentage of the total transactions.

To address this challenge, Random Undersampling was applied before model training.

### Dataset Source

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## ⚙️ Machine Learning Pipeline

### 1. Data Collection & Loading

* Load transaction dataset using Pandas

### 2. Exploratory Data Analysis (EDA)

* View dataset structure
* Analyze transaction distributions
* Study fraud vs legitimate transaction patterns

### 3. Data Cleaning

* Missing value verification
* Dataset validation

### 4. Class Distribution Analysis

* Separate legitimate and fraudulent transactions
* Analyze imbalance ratio

### 5. Dataset Balancing

* Random undersampling of legitimate transactions
* Creation of balanced dataset

### 6. Feature Engineering

* Feature and target separation

### 7. Model Training

* Train-Test Split with Stratification
* Logistic Regression model training

### 8. Model Evaluation

* Accuracy calculation on training and testing data

### 9. Model Deployment

* Save trained model using Joblib (.pkl)
* Deploy through Streamlit dashboard

---

## 📈 Model Performance

| Dataset       | Accuracy |
| ------------- | -------- |
| Training Data | 95.4%    |
| Testing Data  | 93.4%    |

The close training and testing accuracy scores indicate minimal overfitting and good generalization performance.

---

## 🖥️ Application Dashboard

The deployed Streamlit application provides:

### System Metrics

* System Status
* Model Status
* Security Level

### Transaction Inputs

* Transaction Time
* Transaction Amount

### Risk Analysis

* Fraud Probability Score
* Risk Distribution Visualization
* Transaction Classification
* Action Recommendation

### Recommendation Logic

| Fraud Probability | Suggested Action    |
| ----------------- | ------------------- |
| < 40%             | Approve Transaction |
| 40% - 70%         | Send for Review     |
| > 70%             | Block Transaction   |

---

## ▶️ Installation & Usage

### Clone Repository
git clone https://github.com/tarzsardana-ship-it/Credit-Card-Fraud-Detection.git

### Move to Project Directory
cd Credit-Card-Fraud-Detection

### Install Required Libraries
pip install -r requirements.txt

### Run Streamlit App
streamlit run app.py

---

## 💡 Business Impact

This system demonstrates how Machine Learning can assist financial institutions by:

* Detecting suspicious transactions automatically
* Reducing financial fraud losses
* Supporting real-time decision making
* Improving customer transaction security
* Assisting fraud investigation processes

---

## 📚 Learning Outcomes

* Data Preprocessing
* Exploratory Data Analysis (EDA)
* Handling Imbalanced Datasets
* Machine Learning Model Development
* Logistic Regression Implementation
* Model Evaluation and Validation
* Streamlit Deployment
* Real-Time Prediction Systems

---

## 🔮 Future Improvements

* Random Forest & XGBoost Models
* Advanced Feature Engineering
* Real-Time Banking API Integration
* Fraud Alert Notification System
* Model Comparison Dashboard
* Cloud Database Connectivity
