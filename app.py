import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import pmdarima as pm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from io import StringIO

# Streamlit page config
st.set_page_config(page_title="Hybrid AI-Driven Predictive Analytics", layout="wide")
st.title("Hybrid AI-Driven Predictive Analytics")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload CSV with 'Date' and 'Sales' columns", type="csv")

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)

    # Preprocess data
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    df['Sales'] = df['Sales'].fillna(df['Sales'].mean())
    st.write("### Data Preview:", df.head())

    # --- Train / Test Split ---
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    y_train = train_data['Sales']
    y_test = test_data['Sales']

    # --- SARIMA Model ---
    sarima_model = pm.auto_arima(y_train, seasonal=False, stepwise=True, suppress_warnings=True)
    sarima_forecast = sarima_model.predict(n_periods=len(y_test))
    sarima_forecast = pd.Series(sarima_forecast, index=y_test.index)

    # --- XGBoost Model ---
    def create_features(series, lags=3):
        df_feat = pd.DataFrame({'Sales': series})
        for i in range(1, lags + 1):
            df_feat[f'lag_{i}'] = df_feat['Sales'].shift(i)
        return df_feat.dropna()

    df_feat = create_features(df['Sales'])
    X = df_feat.drop('Sales', axis=1)
    y = df_feat['Sales']
    X_train = X.loc[X.index.intersection(y_train.index)]
    y_train_xgb = y.loc[X_train.index]
    X_test = X.loc[X.index.intersection(y_test.index)]

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train_xgb)
    xgb_forecast = pd.Series(xgb_model.predict(X_test), index=X_test.index)

    # --- LSTM Model ---
    scaler = MinMaxScaler()
    sales_scaled = scaler.fit_transform(df[['Sales']])

    def create_lstm_data(data, steps=3):
        X, y = [], []
        for i in range(len(data) - steps):
            X.append(data[i:i+steps, 0])
            y.append(data[i+steps, 0])
        return np.array(X), np.array(y)

    n_steps = 3
    X_all, y_all = create_lstm_data(sales_scaled, n_steps)
    train_len = len(y_train) - n_steps
    X_train_lstm = X_all[:train_len]
    X_test_lstm = X_all[train_len:train_len + len(y_test)]
    y_train_lstm = y_all[:train_len]
    y_test_lstm = y_all[train_len:train_len + len(y_test)]

    X_train_lstm = X_train_lstm.reshape(-1, n_steps, 1)
    X_test_lstm = X_test_lstm.reshape(-1, n_steps, 1)

    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, 1)),
        Dense(1)
    ])

    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, verbose=0)

    lstm_forecast_scaled = lstm_model.predict(X_test_lstm)
    lstm_forecast = pd.Series(scaler.inverse_transform(lstm_forecast_scaled).flatten(), index=y_test.index)

    # --- Ensemble Forecast ---
    min_len = min(len(sarima_forecast), len(xgb_forecast), len(lstm_forecast))
    ensemble_forecast = (sarima_forecast[-min_len:].values + xgb_forecast[-min_len:].values + lstm_forecast[-min_len:].values) / 3
    ensemble_forecast = pd.Series(ensemble_forecast, index=y_test.index[-min_len:])

    # --- Plot Results ---
    st.subheader("Predictions vs Actuals")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index[-min_len:], y_test[-min_len:], label='Actual', marker='o')
    ax.plot(df.index[-min_len:], sarima_forecast[-min_len:], label='SARIMA', linestyle='--')
    ax.plot(df.index[-min_len:], xgb_forecast[-min_len:], label='XGBoost', linestyle='--')
    ax.plot(df.index[-min_len:], lstm_forecast[-min_len:], label='LSTM', linestyle='--')
    ax.plot(df.index[-min_len:], ensemble_forecast, label='Ensemble', color='black', linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)

    # --- Future Prediction (Next 6 Months) ---
    st.subheader("Future Prediction (Next 6 Months)")
    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=6, freq='MS')

    # SARIMA Future Forecast
    n_periods_full = len(y_test) + 6
    sarima_full_pred = sarima_model.predict(n_periods=n_periods_full)
    sarima_future = pd.Series(sarima_full_pred[-6:], index=future_dates)

    # XGBoost Future Forecast
    xgb_future = []
    last_3 = df['Sales'].tail(3).tolist()
    for _ in range(6):
        X_input = pd.DataFrame([last_3], columns=[f'lag_{i}' for i in range(1, 4)])
        pred = xgb_model.predict(X_input)[0]
        xgb_future.append(pred)
        last_3.pop(0)
        last_3.append(pred)

    xgb_future = pd.Series(xgb_future, index=future_dates)

    # LSTM Future Forecast
    lstm_future = []
    last_steps = sales_scaled[-n_steps:].reshape(1, n_steps, 1)
    for _ in range(6):
        pred_scaled = lstm_model.predict(last_steps, verbose=0)[0]
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        lstm_future.append(pred)
        last_steps = np.append(last_steps[:, 1:, :], pred_scaled.reshape(1, 1, 1), axis=1)

    lstm_future = pd.Series(lstm_future, index=future_dates)

    # Ensemble Future Forecast
    ensemble_future = (sarima_future + xgb_future + lstm_future) / 3

    # --- Plot Future Forecast ---
    st.subheader("Future Forecast")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(df.index, df['Sales'], label='Historical Sales', color='blue')
    ax2.plot(future_dates, sarima_future, label='SARIMA Forecast', linestyle=':', color='green')
    ax2.plot(future_dates, xgb_future, label='XGBoost Forecast', linestyle=':', color='red')
    ax2.plot(future_dates, lstm_future, label='LSTM Forecast', linestyle=':', color='purple')
    ax2.plot(future_dates, ensemble_future, label='Ensemble Forecast', linestyle='-', color='black', linewidth=2)
    ax2.set_title('Sales Forecast for Next 6 Months')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Sales')
    ax2.legend()
    st.pyplot(fig2)

else:
    st.info("Upload a CSV file to start predictions.")
