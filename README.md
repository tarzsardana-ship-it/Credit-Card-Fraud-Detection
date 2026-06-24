# Credit Card Fraud Detection using Machine Learning
Credit card fraud is one of the major challenges faced by financial institutions. This project applies Machine Learning techniques to identify fraudulent transactions and distinguish them from legitimate ones.

The dataset is highly imbalanced, with fraudulent transactions representing only a small fraction of all transactions. To address this issue, undersampling was used to create a balanced dataset before training the model.
A Logistic Regression classifier was developed to predict whether a transaction is fraudulent or legitimate.

## Key Highlights
✔ Data Exploration and Analysis
✔ Handling Imbalanced Dataset using Undersampling
✔ Feature and Target Separation
✔ Train-Test Split with Stratification
✔ Logistic Regression Model Implementation
✔ Performance Evaluation and Accuracy Analysis

## Tech Stack
* Python
* NumPy
* Pandas
* Scikit-Learn

## Machine Learning Pipeline
1. Data Collection and Loading
2. Exploratory Data Analysis (EDA)
3. Missing Value Verification
4. Fraudulent vs Legitimate Transaction Analysis
5. Dataset Balancing through Random Undersampling
6. Model Training using Logistic Regression
7. Model Evaluation on Training and Testing Data

## Installation
Clone the repository:
https://github.com/tarzsardana-ship-it/Credit-Card-Fraud-Detection-ML.git

Install dependencies:
pip install -r requirements.txt

Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  
## Results
| Dataset       | Accuracy |
| ------------- | -------- |
| Training Data | 95.4%      |
| Testing Data  | 93.4%      |

The model demonstrates strong predictive performance with minimal overfitting, as evidenced by the close training and testing accuracy scores.

## Learning Outcomes
* Data preprocessing
* Handling imbalanced datasets
* Machine Learning model development
* Model evaluation and validation
* Python-based data analysis
