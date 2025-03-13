import streamlit as st

# Title of the app
st.title(" 🎈Regional Malaria Cases Forecasting Models🎈")
st.info("Forecast malaria cases for Juba, Yei, and Wau based on rainfall and temperature using various models.")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write(data)

    # Preprocess uploaded data
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

