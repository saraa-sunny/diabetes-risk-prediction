# Diabetes risk prediction

A simple, interactive Streamlit web app that predicts the risk of diabetes based on basic health indicators like glucose level, BMI, age, and insulin level.
The model is built using Logistic Regression from scikit-learn.

## Problem statement
Diabetes is a condition that happens when your blood sugar (glucose) is too high. It develops when your pancreas doesn’t make enough insulin or any at all, or when your body isn’t responding to the effects of insulin properly.

## 🧬 Dataset
* Source: Pima Indians Diabetes Database (Kaggle)
* Samples: 768
* Features: 8 independent variables (Glucose, BMI, Age, Insulin, etc.)
* Target: Outcome — 1 indicates diabetes, 0 indicates no diabetes

## 💻 Dependencies
This project requires Python 3.x and the following libraries:
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* streamlit

## 📊 How It Works

1. Data Cleaning

Replaces missing or zero values in key columns (Glucose, BMI, etc.) with the median.

2. Feature Selection & Scaling

Uses only the most correlated features (Glucose, BMI, Age, Insulin).
Scales numeric values with StandardScaler.

3. Model Training

Trains a LogisticRegression classifier using an 80/20 train-test split.

4. Risk Prediction

Takes user input, scales it, predicts diabetes class (0 or 1)

## 📈 Future Improvements

* Add ROC curve and accuracy display
* Allow dataset upload & retraining from the UI
* Include additional models (Random Forest, XGBoost)
* Deploy on Streamlit Cloud
