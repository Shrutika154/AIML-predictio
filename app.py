import streamlit as st
import numpy as np
import joblib as jb

st.title("Vehicle Mileage Impact Prediction 🚗")

st.write("Predict whether a vehicle has positive or negative mileage impact.")

# Load models
lr_model = jb.load("classifier/LR/mpg_lr_model.pkl")
knn_model = jb.load("classifier/KNN/mpg_knn_model.pkl")
scaler = jb.load("classifier/scaler.pkl")

# Model selection
model_choice = st.selectbox(
    "Choose Model",
    ["Logistic Regression", "KNN"]
)

# User inputs
cyl = st.number_input("Number of cylinders (1–8)", 1, 8)
dis = st.number_input("Displacement (30–600)", 30.0, 600.0)
hp = st.number_input("Horsepower (20–300)", 20.0, 300.0)
wt = st.number_input("Weight (1200–5500)", 1200, 5500)
acc = st.number_input("Acceleration (5–28)", 5.0, 28.0)
model_year = st.number_input("Model year (60–90)", 60, 90)
org = st.selectbox("Origin (1=USA, 2=EUR, 3=JAP)", [1, 2, 3])

# Prepare input
new_data = np.array([[cyl, dis, hp, wt, acc, model_year, org]])
new_data_scaled = scaler.transform(new_data)

# Prediction
if st.button("Predict Mileage Impact"):
    if model_choice == "Logistic Regression":
        prediction = lr_model.predict(new_data_scaled)
    else:
        prediction = knn_model.predict(new_data_scaled)

    if prediction[0] == 1:
        st.success("✅ Positive mileage impact (less CO₂ emission)")
    else:
        st.error("❌ Negative mileage impact (more CO₂ emission)")
