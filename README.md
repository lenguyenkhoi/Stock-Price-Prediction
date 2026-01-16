# 📈 Stock Price Prediction

## 📌 Overview
This project focuses on **stock price prediction** using historical data from the **S&P 500 index**.  
We implement and compare three different approaches:

- **ARIMA** – Statistical time series model  
- **Random Forest** – Machine Learning (Ensemble Learning)  
- **LSTM** – Deep Learning (Recurrent Neural Network)

The goal is to evaluate their performance and identify the most suitable model for forecasting stock prices in volatile financial markets.

---
Link demo: https://stock-price-prediction-demo.streamlit.app/
---
## 🎯 Objectives
- Predict future stock prices using historical market data  
- Compare traditional statistical models with machine learning and deep learning models  
- Evaluate models using quantitative metrics  
- Analyze model accuracy, stability, and generalization ability  

---

## 📊 Dataset
- **Source:** S&P 500 historical stock data  
- **Frequency:** Daily trading data  
- **Main features:**
  - Open
  - High
  - Low
  - Close (target variable)
  - Volume

### Feature Engineering
- Simple & Exponential Moving Average (SMA, EMA)  
- RSI (Relative Strength Index)  
- MACD  
- Bollinger Bands  
- Lag features (previous day prices)

---

## 🔄 Data Processing
1. Remove non-trading days (weekends, holidays)  
2. Handle missing values using forward fill  
3. Normalize data using Min-Max Scaling  
4. Split data based on time order (no shuffling)  

---

## 🧠 Models
### ARIMA
- Linear statistical time series model  
- Requires stationary data  
- Suitable for short-term forecasting  

### Random Forest
- Ensemble learning model using multiple decision trees  
- Captures nonlinear relationships  
- Requires lag features to model time dependency  

### LSTM
- Recurrent Neural Network designed for time series  
- Captures long-term dependencies  
- Effective in volatile and nonlinear markets  

---

## 📈 Results

| Model | Avg. MAPE | Avg. RMSE | Performance |
|------|----------|----------|-------------|
| ARIMA | 32.24% | 40.65 USD | Poor |
| Random Forest | 3.79% | 7.29 USD | Very Good |
| LSTM | **3.09%** | **4.84 USD** | **Excellent** |

---

## 🏆 Conclusion
- **LSTM** achieved the best performance with the lowest prediction error  
- **Random Forest** performed well for large-scale portfolio analysis  
- **ARIMA** showed significant limitations in handling nonlinear market behavior  

---

## 🚀 Future Improvements
- Integrate sentiment analysis from financial news and social media  
- Develop hybrid models (CNN-LSTM, ARIMA-LSTM)  
- Apply automated hyperparameter optimization  
- Deploy a real-time prediction system  

---

## 🛠️ Technologies
- Python  
- Pandas, NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- Statsmodels  
