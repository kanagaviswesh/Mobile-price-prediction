import streamlit as st
import pickle
import numpy as np

# Load the saved Linear Regression model
with open('PICKLE.sav', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict EMISSION using the loaded model
def predict_price(Storage,RAM,Screen_Size,Battery_Capacity,Brand_Apple,Brand_Google,Brand_OnePlus,Brand_Samsung):
    features = np.array([Storage,RAM,Screen_Size,Battery_Capacity,Brand_Apple,Brand_Google,Brand_OnePlus,Brand_Samsung])
    features = features.reshape(1,-1)
    price = model.predict(features)
    return price[0]

# Streamlit UI
st.title('PRICE Prediction')
st.write("""
## Input Features
Enter the values for the input features to predict EMISSION.
""")

# Input fields for user 
Storage = st.number_input('Storage')
RAM = st.number_input('RAM')
Screen_Size = st.number_input('Screen_Size')
Battery_Capacity = st.number_input('Battery_capacity')
Brand_Apple = st.number_input('Brand_Apple')
Brand_Google= st.number_input('Brand_Google')
Brand_OnePlus = st.number_input('Brand_Oneplus')
Brand_Samsung = st.number_input('Brand_Samsung')

# Prediction button
if st.button('Predict'):
    # Predict EMISSION
    price_prediction = predict_price(Storage,RAM,Screen_Size,Battery_Capacity,Brand_Apple,Brand_Google,Brand_OnePlus,Brand_Samsung)
    st.write(f"Predicted SELLING PRICE: {price_prediction}")