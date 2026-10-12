import streamlit as st
import pickle
import numpy as np
import pandas as pd

with open('logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Diabetes Prediction")

st.write("Enter the patient's details to predict the diabetes outcome.")

pregnancies = st.number_input("Pregnancies", min_value=0, value=1)
glucose = st.number_input("Glucose", min_value=0, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, value=20)
insulin = st.number_input("Insulin", min_value=0, value=80)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.5
)
age = st.number_input("Age", min_value=1, value=30)

if st.button("Predict"):

    input_data = pd.DataFrame([[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
]], columns=[
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("The model predicts that the person may have diabetes.")
    else:
        st.success("The model predicts that the person may not have diabetes.")