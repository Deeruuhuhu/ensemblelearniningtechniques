import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("🌲 Purchase Prediction App")

age = st.number_input("Enter Age")
salary = st.number_input("Enter Salary")

if st.button("Predict"):
    result = model.predict([[age, salary]])

    if result[0] == 1:
        st.success("Will Purchase")
    else:
        st.error("Will Not Purchase")
