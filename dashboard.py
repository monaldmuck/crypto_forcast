import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load all forcasts CSVs from forecasts folder
forecast_files = [f for f in os.listdir('forecasts') if f.endswith('_forcasts.csv')]
summary_df = pd.read_csv('forecasts/summary_metrics.csv')

# Sidebar for asset selection 
st.sidebar.title("Select an Asset")
symbol = st.sidebar.selectbox("Coose an asset to visualize", [f.replace('_forcasts.csv', '') for f in forecast_files])

# Display category info
forcast_path = f'forecasts/{symbol}_forcasts.csv'
df = pd.read_csv(forcast_path)

#Page title
st.title(f"{symbol} Price Forcast")
st.markdown(f"### Category: {category}")

# Line plot of the true vs predicted 
st.subheader("📈 True vs Predicted Price")
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df['Date'], df['True Price'], label='True Price',linewidth=2)
ax.plot(df['Date'], df['Predicted Price'], label='Predicted Price', linestyle='--', linewidth=2)
ax.set_xlabel('Date')
ax.set_ylabel('Price(USD)')
ax.legend()
st.pyplot(fig)

# Evaluation metrics
st.subheader("🧮 Evaluation Metrics")
metrics = summary_df.loc[summary_df.index == symbol].drop(columns=['Category',]).T
st.table(metrics.style.format({"Value:":"{:.4f}"}))

# Comparison plot 
st.subheader("📊 Compare All Assets by MAPE")
bar_fig, bar_ax = plt.subplots
summary_df_sorted = summary_df.sort_values('mape')
bar_ax.barh(summary_df.sorted.index, summary_df_sorted['mape']* 100)
bar_ax.set_xlabel('MAPE (%)')
bar_ax.set_title('Model Accuracy (Lower is Better)')
st.pyplot(bar_fig)