import streamlit as st
import seaborn as sns
import pickle
import torch
import joblib
import json
import matplotlib.pyplot as plt
from prophet import Prophet
from neuralprophet import NeuralProphet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from keras.models import model_from_json
from fbprophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from neuralprophet import NeuralProphet, set_log_level
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_predict

# Title of the app
st.title(" 🎈Regional Malaria Cases Forecasting Models🎈")
st.write("Forecast malaria cases for Juba, Yei, and Wau based on rainfall and temperature using various models.")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write(data)

    # Preprocess uploaded data
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

