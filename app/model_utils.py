# app/model_utils.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def load_and_clean_data(csv_path='data/diabetes.csv'):
    """
    Loads the diabetes dataset and performs basic cleaning.

    Steps:
    - Reads the CSV file
    - Replaces 0 values in key numeric columns with their column median

    Args:
        csv_path (str): Path to the dataset CSV file

    Returns:
        df (DataFrame): Cleaned pandas DataFrame
    """
    df = pd.read_csv(csv_path)

    # Replace invalid 0s with median for specific columns
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[col] = df[col].replace(0, df[col].median())
    return df


def prepare_data(df):
    """
    Selects features, scales them, and splits the dataset for training/testing.

    Args:
        df (DataFrame): Cleaned input DataFrame

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """

    # Select only the most important features (based on correlation/interpretability)
    selected_features = ['Glucose', 'BMI', 'Age', 'Insulin']
    X = df[selected_features]
    y = df['Outcome']

    # Initialize and fit the StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler


def train_model(X_train, y_train):
    """
    Trains a Logistic Regression model using the training data.

    Args:
        X_train (ndarray): Scaled training features
        y_train (Series): Training labels

    Returns:
        model (LogisticRegression): Trained model
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def predict_risk(model, scaler, input_data):
    """
    Predicts the diabetes risk for a given input.

    Steps:
    - Converts the input data (dict) into a DataFrame
    - Scales it using the trained StandardScaler
    - Predicts both class (0 or 1) and probability

    Args:
        model (LogisticRegression): Trained model
        scaler (StandardScaler): Fitted scaler from training
        input_data (dict): User input data with required feature names

    Returns:
        risk_class (ndarray): Predicted class (0 or 1)
    """
    #Convert input dict into DataFrame
    df = pd.DataFrame([input_data])
    
    # Apply same scaling as training data
    df_scaled = scaler.transform(df)

    # Predict class and probability
    risk_class = model.predict(df_scaled)
    return risk_class