import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import joblib

kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Welcome in Customer Segmentation App')
st.write('Enter customer details to predict the segment.')

age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
score = st.slider("Spending Score (1–100)", min_value=1, max_value=100, value=50)




if st.button("Predict Cluster"):
    input_data = np.array([[age, income, score]])
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]
    
    st.success(f"📊 This customer belongs to Cluster #{cluster}")