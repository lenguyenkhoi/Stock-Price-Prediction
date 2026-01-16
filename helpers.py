import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import streamlit as st

# EDA
def plot_all_histograms(data, title_prefix=""):
    num_cols = data.select_dtypes(include=[np.number]).columns
    n_cols = 3
    n_rows = math.ceil(len(num_cols) / n_cols)

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(data[col], kde=True, bins=30)
        plt.title(f"{title_prefix}{col}")
        plt.xlabel("")
        plt.ylabel("")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
# Feature Engineering
def add_technical_indicators(df):
  df['SMA_10'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(10).mean())
  df['EMA_10'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=10).mean())

  delta = df.groupby('Symbol')['Close'].diff()
  gain = delta.clip(lower=0)
  loss = -delta.clip(upper=0)

  avg_gain = gain.groupby(df['Symbol']).transform(lambda x: x.rolling(14).mean())
  avg_loss = loss.groupby(df['Symbol']).transform(lambda x: x.rolling(14).mean())

  rs = avg_gain / avg_loss
  df['RSI'] = 100 - (100 / (1 + rs))

  df['ROC'] = df.groupby('Symbol')['Close'].pct_change(5)
  df['OBV'] = df.groupby('Symbol')['Volume'].cumsum()

  return df

