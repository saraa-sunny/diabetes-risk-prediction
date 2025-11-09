# app/main_app.py

import streamlit as st
from model_utils import load_and_clean_data,prepare_data, train_model, predict_risk

@st.cache_resource
def load_model_and_scaler():
    """
    Loads and preprocesses the diabetes dataset, 
    trains the logistic regression model, 
    and returns the model and the fitted scaler.

    Returns:
        model (sklearn model): Trained Logistic Regression model
        scaler (StandardScaler): Fitted scaler for preprocessing input data
    """
    df = load_and_clean_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    model = train_model(X_train, y_train)
    return model, scaler

# Load the model & scaler once
model, scaler = load_model_and_scaler()


# streamlit ui 
st.title("🩺 Diabetes Risk Prediction Tool")
st.markdown("""
This tool predicts the **risk of diabetes** based on basic health information.  
""")

st.subheader("Enter health data:")

# user input 
glucose = st.number_input("Glucose",min_value=0,max_value=1000,value=120)
bmi= st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
age=st.number_input("Age", min_value=0, max_value=120, value=30)
insulin = st.number_input("Insulin Level", min_value=1, max_value=850, value=80)


if st.button("Predict Risk"):
    # Combine user inputs into a dictionary
    input_data = {'Glucose': glucose, 'BMI': bmi, 'Age': age, 'Insulin': insulin}
    
    # Predict diabetes risk
    pred_class = predict_risk(model, scaler, input_data)

    # Display results
    if pred_class == 1:
        st.error("High risk of diabetes !!")
    else:
        st.success("Low risk of diabetes.")