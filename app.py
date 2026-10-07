
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

st.title("🏠 House Price Prediction App")

st.write("Enter the house features to predict its price:")

# Input fields
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1.0, 5.0, 2.0)
sqft_living = st.number_input("Sqft Living", 400, 10000, 1500)
sqft_lot = st.number_input("Sqft Lot", 500, 50000, 5000)
floors = st.number_input("Floors", 1.0, 3.5, 1.0)
waterfront = st.selectbox("Waterfront (1=Yes, 0=No)", [0, 1])
view = st.slider("View", 0, 4, 0)
condition = st.slider("Condition", 1, 5, 3)
grade = st.slider("Grade", 1, 13, 7)
sqft_above = st.number_input("Sqft Above", 400, 8000, 1200)
sqft_basement = st.number_input("Sqft Basement", 0, 4000, 500)
zipcode = st.number_input("Zipcode", 98000, 98200, 98103)
lat = st.number_input("Latitude", 47.0, 48.0, 47.5)
long = st.number_input("Longitude", -123.0, -121.0, -122.2)
sqft_living15 = st.number_input("Sqft Living 15", 400, 7000, 1500)
sqft_lot15 = st.number_input("Sqft Lot 15", 500, 50000, 5000)
house_age = st.number_input("House Age", 0, 120, 30)
recent_renovation = st.selectbox("Recently Renovated (1=Yes, 0=No)", [0, 1])

# Prepare input
features = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, 
                      condition, grade, sqft_above, sqft_basement, zipcode, lat, long,
                      sqft_living15, sqft_lot15, house_age, recent_renovation]])

features_scaled = scaler.transform(features)
pred = model.predict(features_scaled)
pred_price = np.expm1(pred[0])

st.success(f"🏡 Predicted House Price: ${pred_price:,.2f}")
