import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Loading saved components
# model = joblib.load("rf_model.pkl")
# label_encoder = joblib.load("label_encoder.pkl")
# scaler = joblib.load("scaler.pkl")
# ir_model = joblib.load("rf_ir.pkl")
# gdp_model = joblib.load("rf_gdp.pkl")
rf_model_reg = joblib.load("rf_model_reg.pkl")

st.title("Financial and Macroeconomic Indicator Simulator")
# st.write("Select the values and click 'Predict Stock Index' to predict the stock index")
st.write("Select the values and click 'Predict Close Price' to predict the close price")

features_cp = [
            st.slider('Trading Volume: ',1600000,1000000000),                    
            st.slider('GDP Growth (%): ',0.0,10.0), 
            st.slider('Inflation Rate (%): ',0.0,10.0),
            st.slider('Unemployment Rate (%): ',0.0,15.0),
            st.slider('Interest Rate (%): ',0.0,10.0),
            st.slider('Consumer Confidence Index: ',50,120),
            st.slider('Government Debt (Billion USD): ',500,30000),
            st.slider('Corporate Profits (Billion USD): ',100,5000),
            st.slider('Forex USD/EUR: ',0.0,1.5),
            st.slider('Forex USD/JPY: ',80.0,150.0),
            st.slider('Crude Oil Price (USD per Barrel): ',20.0,150.0),
            st.slider('Gold Price (USD per Ounce): ',800.0,2500.0),
            st.slider('Real Estate Index: ',100.0,500.0),
            st.slider('Retail Sales (Billion USD): ',100,10000),
            st.slider('Bankruptcy Rate (%): ',0.0,10.0),
            st.slider('Mergers & Acquisitions Deals: ',0,50),
            st.slider('Venture Capital Funding (Billion USD): ',0.0,100.0),
            st.slider('Consumer Spending (Billion USD: ',100,15000)
            ]

#Converting and reshaping
input_array = np.array(features_cp).reshape(1, -1)


# features_si = [
#             st.slider('Trading Volume: ',1600000,1000000000),                  
#             st.slider('GDP Growth (%): ',0.0,10.0),
#             st.slider('Inflation Rate (%): ',0.0,10.0),
#             st.slider('Unemployment Rate (%): ',0.0,15.0),
#             st.slider('Interest Rate (%): ',0.0,10.0),
#             st.slider('Consumer Confidence Index: ',50,120),
#             st.slider('Government Debt (Billion USD): ',500,30000),
#             st.slider('Corporate Profits (Billion USD): ',100,5000),
#             st.slider('Forex USD/EUR: ',0.0,1.5),
#             st.slider('Forex USD/JPY: ',80.0,150.0),
#             st.slider('Crude Oil Price (USD per Barrel): ',20.0,150.0),
#             st.slider('Gold Price (USD per Ounce): ',800.0,2500.0),
#             st.slider('Real Estate Index: ',100.0,500.0),
#             st.slider('Retail Sales (Billion USD): ',100,10000),
#             st.slider('Bankruptcy Rate (%): ',0.0,10.0),
#             st.slider('Mergers & Acquisitions Deals: ',0,50),
#             st.slider('Venture Capital Funding (Billion USD): ',0.0,100.0),
#             st.slider('Consumer Spending (Billion USD: ',100,15000)
#             ]

# #Converting and scaling
# input_array = np.array(features_si).reshape(1, -1)
# input_scaled = scaler.transform(input_array)


# Making predictions
if st.button('Predict Close Price'):
    y_pred = rf_model_reg.predict(input_array)
    st.write(f'Predicted Close Price: {y_pred[0]:.2f}')

# elif st.button('Predict Stock Index'):
#     prediction = model.predict(input_scaled)
#     prediction_label = label_encoder.inverse_transform(prediction)
#     st.write(f'Predicted output: {prediction_label[0]}')
