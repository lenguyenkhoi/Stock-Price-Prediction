import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import kagglehub
from kagglehub import KaggleDatasetAdapter
from helpers import *
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("Stock Price Prediction")
st.write("This app predicts stock prices focuses on building and comparing the performance of three models—ARIMA, LSTM, and Random Forest—for stock price prediction. By extracting features from historical financial data and evaluating the models using quantitative metrics, the research aims to identify the most suitable model for forecasting stock prices based on S&P 500 data.")

st.header("S&P 500 Stock Data")

file_path = "sp500_stocks.csv"

# Load the latest version
data = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS,"andrewmvd/sp-500-stocks",file_path,)
#Kiếm tra dữ liệu
# print("First 5 records:", data.head())
# print(data.info)
# data = pd.read_csv('data/sp500_stocks.csv')
st.dataframe(data.head(50))

with st.expander("Show Summary Statistics"):
    st.subheader("Summary Statistics")
    st.write(data.describe())
    st.subheader("Data Types")
    st.write(data.dtypes)

st.header("Preprocess the data")
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(['Symbol', 'Date']).reset_index(drop=True)

data.set_index('Date', inplace=True)
data.dropna(inplace=True)
st.write(data.head())
with st.expander("Show Summary Statistics"):
    st.subheader("Summary Statistics")
    st.write(data.describe())
    st.subheader("Data Types")
    st.write(data.dtypes)
    st.subheader("Missing Values")
    st.write(data.isnull().sum())


# st.header("Histograms of Numerical Features")
# plot_all_histograms(data, title_prefix="Distribution of ")

st.header("Feature Engineering")
st.write("""
         Raw price data alone is often insufficient for models to effectively capture market conditions. Therefore, this study incorporates technical indicators to enrich the input features for the models:

        - Moving Averages (SMA – Simple Moving Average, EMA – Exponential Moving Average): Used to identify price trends.
        - RSI (Relative Strength Index): Evaluates overbought and oversold market conditions.
        - MACD (Moving Average Convergence Divergence): Detects changes in trend strength and momentum.
        - Bollinger Bands: Measure price volatility based on the standard deviation around a moving average.
        - Lag features: Provide historical price information from previous days (t-1, t-2), offering temporal context for the models.
        These features enable the models to better understand the underlying dynamics of market movements.
         """)

data_feats = add_technical_indicators(data)
forecast_horizon = 5  # 5 ngày

data_feats['Future_Close'] = data_feats.groupby('Symbol')['Close'].shift(-forecast_horizon)
data_feats = data_feats.dropna().reset_index(drop=True)

# data_feats = pd.read_csv('data/sp500_stocks_with_features.csv')
st.write(data_feats.head(50))
st.header("Histograms of Numerical Features")
plot_all_histograms(data_feats, title_prefix="Distribution of ")

st.header("Train model")
st.subheader("Evaluation 30 stocks")
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Random Forest Regressor")
    st.subheader("Results")
    st.image("image/RF_30stock.png", caption="Random Forest Regressor Results")
    st.write("This result was obtained from the RandomForestRegressor.ipynb")
with col2:
    st.subheader("LSTM Model")
    st.subheader("Results")
    st.image("image/LSTM_30stock.png", caption="LSTM Model Results")
    st.write("This result was obtained from the LSTM.ipynb")
with col3:
    st.subheader("ARIMA Model")
    st.subheader("Results")
    st.image("image/ARIMA_30stock.png", caption="ARIMA Model Results")
    st.write("This result was obtained from the ARIMA.ipynb")
    
st.header("Conclusion")
st.write(""" 
        **Stability of LSTM:**
        - With a MAPE of only **3.09%** and an extremely low RMSE of 4.84 USD, this model demonstrates a highly concentrated and minimal error distribution. This result confirms that LSTM effectively filters out noise, enabling predictions to closely track actual price movements without significant deviations, even during periods of strong market volatility.
        - With the lowest RMSE, LSTM is the most effective model for predicting precise price values, enabling investors to accurately determine stop-loss and take-profit levels.
        
        **Generalization of Random Forest:**
        - Achieving a MAPE of **3.79%** across a portfolio of 30 stocks is a highly impressive result. However, an RMSE of 7.29 USD—higher than that of LSTM—indicates that although the relative percentage error remains low, Random Forest can still exhibit notable absolute deviations for stocks with higher price levels.
        - Random Forest serves as an ideal tool for scanning the entire S&P 500 market to identify stocks with strong potential growth trends.
        
        **Failure of ARIMA:**
        - An average MAPE of **32.24%**, with some stocks reaching an extreme **123.31%**, clearly indicates that ARIMA lacks robustness when facing nonlinear characteristics. Consequently, it fails to provide reliable forecasts across a diversified stock portfolio.
        - This model should primarily be used as a baseline or validation method to highlight the superiority of modern AI-based techniques, rather than being applied independently for investment decision-making.
         """ )

st.subheader("Visual Comparison of Model Performance on GOOG Stock")
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Random Forest Regressor on GOOG")
    st.image("image/RF_GOOG.png")
    st.image("chart/GOOG_rf.png", caption="Random Forest Regressor on GOOG")
    st.write("This chart was obtained from the RandomForestRegressor.ipynb")
with col2:
    st.subheader("LSTM Model on GOOG")
    st.image("image/LSTM_GOOG.png")
    st.image("chart/GOOG_LSTM.png", caption="LSTM Model on GOOG")
    st.write("This chart was obtained from the LSTM.ipynb")
with col3:
    st.subheader("ARIMA Model on GOOG")
    st.image("image/ARIMA_GOOG.png")
    st.image("chart/GOOG_ARIMA.png", caption="ARIMA Model on GOOG")
    st.write("This chart was obtained from the ARIMA.ipynb")
    
st.subheader("Concluding based on GOOG Stock charts")
st.write("""
        - The LSTM model's predictions closely follow the actual stock prices, demonstrating its ability to capture complex patterns and trends in the data.
        - The Random Forest model also performs well, but with slightly larger deviations from the actual prices compared to LSTM.
        - The ARIMA model shows significant discrepancies, particularly during volatile market periods, highlighting its limitations in handling nonlinear stock price movements.
        Overall, the visual comparison reinforces the quantitative findings, confirming LSTM as the superior model for stock price prediction in this study.
         """)