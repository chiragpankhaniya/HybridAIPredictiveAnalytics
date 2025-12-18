import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pmdarima import auto_arima
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------
# App Configuration
# -----------------------------------------
st.set_page_config(
    page_title="Hybrid AI Business Forecasting",
    layout="wide"
)

st.title("📊 Hybrid AI Auto-Analytics & Sales Forecasting")
st.markdown("""
Upload your business dataset and get **automatic charts and future predictions**
using **Hybrid AI (SARIMA + XGBoost + LSTM)**.
""")

# -----------------------------------------
# File Upload
# -----------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file (Date, Sales)", type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    st.success("Dataset uploaded successfully!")
    st.dataframe(df.head())

    # -----------------------------------------
    # Visualization
    # -----------------------------------------
    st.subheader("📈 Historical Sales Trend")
    fig = px.line(df, x=df.index, y='Sales')
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------
    # Train-Test Split
    # -----------------------------------------
    train_size = int(len(df) * 0.85)
    train, test = df[:train_size], df[train_size:]

    y_train = train['Sales']
    y_test = test['Sales']

    # -----------------------------------------
    # SARIMA
    # -----------------------------------------
    with st.spinner("Training SARIMA model..."):
        sarima_model = auto_arima(
            y_train,
            seasonal=True,
            m=12,
            suppress_warnings=True
        )
        sarima_pred = sarima_model.predict(n_periods=len(y_test))

    # -----------------------------------------
    # XGBoost
    # -----------------------------------------
    def create_features(data, lags=6):
        df_feat = data.copy()
        for i in range(1, lags+1):
            df_feat[f'lag_{i}'] = df_feat['Sales'].shift(i)
        return df_feat.dropna()

    xgb_data = create_features(df)
    train_xgb = xgb_data.iloc[:train_size-6]
    test_xgb = xgb_data.iloc[train_size-6:]

    X_train, y_train_xgb = train_xgb.drop('Sales', axis=1), train_xgb['Sales']
    X_test, y_test_xgb = test_xgb.drop('Sales', axis=1), test_xgb['Sales']

    with st.spinner("Training XGBoost model..."):
        xgb_model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4
        )
        xgb_model.fit(X_train, y_train_xgb)
        xgb_pred = xgb_model.predict(X_test)

    # -----------------------------------------
    # LSTM
    # -----------------------------------------
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Sales']])

    def create_lstm_data(data, window=6):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X, y = create_lstm_data(scaled)
    X_train_lstm, X_test_lstm = X[:train_size-6], X[train_size-6:]
    y_train_lstm, y_test_lstm = y[:train_size-6], y[train_size-6:]

    with st.spinner("Training LSTM model..."):
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(X_train_lstm.shape[1], 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_lstm, y_train_lstm, epochs=30, verbose=0)

        lstm_pred = model.predict(X_test_lstm)
        lstm_pred = scaler.inverse_transform(lstm_pred)

    # -----------------------------------------
    # Hybrid Ensemble
    # -----------------------------------------
    sarima_pred = sarima_pred[:len(lstm_pred)]
    xgb_pred = xgb_pred[:len(lstm_pred)]

    hybrid_pred = (
        0.3 * sarima_pred +
        0.3 * xgb_pred +
        0.4 * lstm_pred.flatten()
    )

    # -----------------------------------------
    # Evaluation
    # -----------------------------------------
    st.subheader("📌 Model Performance")

    def metrics(y_true, y_pred):
        return (
            mean_absolute_error(y_true, y_pred),
            np.sqrt(mean_squared_error(y_true, y_pred))
        )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("SARIMA MAE", round(metrics(y_test[:len(sarima_pred)], sarima_pred)[0], 2))
    col2.metric("XGBoost MAE", round(metrics(y_test_xgb[:len(xgb_pred)], xgb_pred)[0], 2))
    col3.metric("LSTM MAE", round(metrics(y_test[:len(lstm_pred)], lstm_pred)[0], 2))
    col4.metric("Hybrid MAE", round(metrics(y_test[:len(hybrid_pred)], hybrid_pred)[0], 2))

    # -----------------------------------------
    # Future Forecast
    # -----------------------------------------
    st.subheader("🔮 Future Sales Forecast")

    future_steps = st.slider("Select months to predict", 1, 12, 6)

    sarima_future = sarima_model.predict(n_periods=future_steps)

    last_lags = df['Sales'].values[-6:].tolist()
    xgb_future = []

    for _ in range(future_steps):
        input_data = np.array(last_lags[-6:]).reshape(1, -1)
        pred = xgb_model.predict(input_data)[0]
        xgb_future.append(pred)
        last_lags.append(pred)

    lstm_input = scaled[-6:].reshape(1,6,1)
    lstm_future = []

    for _ in range(future_steps):
        pred = model.predict(lstm_input)
        lstm_future.append(pred[0,0])
        lstm_input = np.append(
            lstm_input[:,1:,:],
            pred.reshape(1,1,1),
            axis=1
        )

    lstm_future = scaler.inverse_transform(
        np.array(lstm_future).reshape(-1,1)
    )

    hybrid_future = (
        0.3 * sarima_future +
        0.3 * np.array(xgb_future) +
        0.4 * lstm_future.flatten()
    )

    future_dates = pd.date_range(
        df.index[-1] + pd.DateOffset(months=1),
        periods=future_steps,
        freq='M'
    )

    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Sales': hybrid_future
    })

    fig2 = px.line(
        pd.concat([df.reset_index(), future_df]),
        x='Date',
        y=['Sales', 'Predicted Sales'],
        title='Hybrid AI Future Forecast'
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.success("Forecast generated successfully!")
