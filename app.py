import streamlit as st
import joblib
from countryinfo import CountryInfo
import numpy as np
import pandas as pd

pipeline = joblib.load("forest_model.pkl")

st.title("Churn Predict")

st.divider()

st.write("Enter values for prediction")

st.divider()

prov_list = CountryInfo("Canada").provinces()

gender = st.selectbox("Enter customer gender", ["Male", "Female"])

education = st.selectbox("Enter customer education", ["High School or Below", "College", "Bachelor", "Master", "Doctor"])

tenure = st.number_input("Enter customer's tenure", min_value = 0, max_value = 500, value = 0)

total_flights = st.number_input("Enter customer total flights", min_value = 0, max_value = 500, value = 0)

enroll = st.selectbox("Enter customer enrollment type", ["2018 Promotion", "Standard"])

province = st.selectbox("Enter customer province", options = prov_list)

points = st.number_input("Enter customer total points", min_value = 0, max_value=1000000, value = 1)

redeemed = st.number_input("Enter redeemed points", min_value = 0, max_value=1000000, value = 0)

clv = st.number_input("Enter customer CLV", min_value=0, max_value=100000, value = 0)

card = st.selectbox("Select membership card", ["Aurora", "Nova", "Star"])

salary = st.number_input("Enter salary", min_value=0, max_value=1000000, value = 0)

marital = st.selectbox("Enter marital status", ["Married", "Single", "Divorced"])

percent_redeemed = round(redeemed/points, 2)*100

distance = st.number_input("Enter distance flown", min_value=0, max_value= 1000000, value = 0)

st.divider()

predict_button = st.button("Predict")

if predict_button:
    X = pd.DataFrame([{
    "tenure": tenure,
    "CLV": clv,
    "Total Flights": total_flights,
    "Total Distance": distance,
    "Total Points": points,
    "Total Redeemed Points": redeemed,
    "Percentage Redeemed (%)": percent_redeemed,
    "Gender": gender,
    "Education": education,
    "Loyalty Card": card,
    "Marital Status": marital,
    "Province": province,
    "Salary": salary,
}])
    
    prediction = pipeline.predict(X)

    predicted = "Churn" if prediction == 1 else "No Churn"

    st.write(f"Prediction: {predicted}")

