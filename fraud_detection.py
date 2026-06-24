#importing libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#version of libraries
print("Numpy Version:", np.__version__)
print("Pandas Version:", pd.__version__)
print("sklearn Version:", sklearn.__version__)
# Load dataset
data = pd.read_csv("creditcard.csv")
# First 5 rows
print("\nFirst 5 Rows")
print(data.head())
# last 5 rows
print("\nLast 5 Rows")
print(data.tail())
# Information about dataset
print("\nDataset Information")
print(data.info())
# Checking for missing values in each column
print("\nMissing Values in Each Column")
print(data.isnull().sum())
# Distribution of legitimate and fraudulent transactions
print("\nDistribution of Legitimate and Fraudulent Transactions")
print(data["Class"].value_counts())
print("This dataset is highly imbalanced, with a very small percentage of fraudulent transactions compared to legitimate ones.")
# seperating data for analysis
legitimate = data[data["Class"] == 0]
fraudulent = data[data["Class"] == 1]
print(legitimate.shape)
print(fraudulent.shape)
#statistical summary of the dataset
print("\nStatistical Summary of the Dataset")
print(legitimate.Amount.describe())
print(fraudulent.Amount.describe())
#comparing the values of legitimate and fraudulent transactions
print("\nComparing the Values of Legitimate and Fraudulent Transactions")
print(data.groupby("Class").mean())
#under-sampling the dataset to balance the classes
print("\nBuild a sample dataset containing similar distribution of legitimate and fraudulent transactions")
legitimate_sample = legitimate.sample(n=492)
balanced_data = pd.concat([legitimate_sample, fraudulent], axis=0)
print(balanced_data.head())
print(balanced_data.tail())
print(balanced_data.shape)
print("\nDistribution of Legitimate and Fraudulent Transactions in the Balanced Dataset")
print(balanced_data["Class"].value_counts())
print("\nComparing the Values of Legitimate and Fraudulent Transactions in the Balanced Dataset")
print(balanced_data.groupby("Class").mean())
#splitting the data into features and target variable
X = balanced_data.drop("Class", axis=1)
Y = balanced_data["Class"]
print("\nFeatures and Target Variable")
print(X.head())
print(Y.head())
#splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print("\nTraining and Testing Sets")
print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape)
#training the logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)
#evaluating the model
print("\nModel Evaluation")
#accuracy on training data
X_train_pred = model.predict(X_train)
print("Accuracy Score on Training Set:", accuracy_score(X_train_pred, Y_train))
#accuracy on testing data
X_test_pred = model.predict(X_test)
print("Accuracy Score on Testing Set:", accuracy_score(X_test_pred, Y_test))