import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import shap
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Attention

warnings.filterwarnings("ignore")

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="Hybrid AI Forecasting",
    layout="wide"
)

st.title("📊 Hybrid AI Auto-Analytics & Sales Forecasting")
st.markdown("""
Upload your business dataset (Date, Sales) and get:
- Automatic visualization
- Forecasting (Hybrid AI)
- Explainable AI (SHAP + Attention)
- Confidence intervals & risk analysis
""")

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file (Date, Sales)", type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    df = df.asfreq('MS')
    df['Sales'] = df['Sales'].interpolate()

    st.success("Dataset uploaded successfully!")
    st.dataframe(df.head())

    # ---------------------------
    # Check dataset size
    # ---------------------------
    data_len = len(df)
    if data_len < 12:
        st.error("Dataset must contain at least 12 rows for forecasting.")
        st.stop()

    # ---------------------------
    # Visualization
    # ---------------------------
    st.subheader("📈 Historical Sales Trend")
    fig = px.line(df, x=df.index, y='Sales', title="Sales History")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # Train-Test Split
    # ---------------------------
    train_size = int(len(df) * 0.85)
    train, test = df[:train_size], df[train_size:]
    y_train = train['Sales']
    y_test = test['Sales']

    # ---------------------------
    # Determine model usage based on dataset size
    # ---------------------------
    use_sarima = data_len >= 24
    use_xgb = data_len >= 18
    use_lstm = data_len >= 36

    # ---------------------------
    # SARIMA
    # ---------------------------
    if use_sarima:
        with st.spinner("Training SARIMA model..."):
            try:
                sarima_model = auto_arima(
                    y_train,
                    seasonal=True,
                    m=12,
                    error_action="ignore",
                    suppress_warnings=True,
                    stepwise=True
                )
                sarima_pred = sarima_model.predict(n_periods=len(y_test))
            except:
                st.warning("SARIMA failed, using naive forecast")
                sarima_pred = np.repeat(y_train.mean(), len(y_test))
    else:
        sarima_pred = np.repeat(y_train.mean(), len(y_test))

    # ---------------------------
    # XGBoost
    # ---------------------------
    def create_features(data, lags=6):
        df_feat = data.copy()
        for i in range(1, lags + 1):
            df_feat[f'lag_{i}'] = df_feat['Sales'].shift(i)
        return df_feat.dropna()

    if use_xgb:
        xgb_data = create_features(df)
        train_xgb = xgb_data.iloc[:train_size - 6]
        test_xgb = xgb_data.iloc[train_size - 6:]

        X_train, y_train_xgb = train_xgb.drop('Sales', axis=1), train_xgb['Sales']
        X_test, y_test_xgb = test_xgb.drop('Sales', axis=1), test_xgb['Sales']

        with st.spinner("Training XGBoost model..."):
            xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
            xgb_model.fit(X_train, y_train_xgb)
            xgb_pred = xgb_model.predict(X_test)
    else:
        xgb_pred = np.repeat(y_train.mean(), len(y_test))

    # ---------------------------
    # LSTM (Optional)
    # ---------------------------
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Sales']])

    def create_lstm_data(data, window=6):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i - window:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X, y_lstm = create_lstm_data(scaled)
    X_train_lstm, X_test_lstm = X[:train_size - 6], X[train_size - 6:]
    y_train_lstm, y_test_lstm = y_lstm[:train_size - 6], y_lstm[train_size - 6:]

    if use_lstm:
        with st.spinner("Training Attention LSTM model..."):
            inputs = Input(shape=(X_train_lstm.shape[1], 1))
            lstm_out = LSTM(64, return_sequences=True)(inputs)
            attention = Attention()([lstm_out, lstm_out])
            dense = Dense(1)(attention[:, -1, :])
            att_model = Model(inputs, dense)
            att_model.compile(optimizer='adam', loss='mse')
            att_model.fit(X_train_lstm, y_train_lstm, epochs=30, verbose=0)
            lstm_pred = att_model.predict(X_test_lstm)
            lstm_pred = scaler.inverse_transform(lstm_pred)
    else:
        lstm_pred = np.repeat(y_train.mean(), len(y_test))

    # ---------------------------
    # Hybrid Ensemble
    # ---------------------------
    preds_list = []
    if use_sarima: preds_list.append(sarima_pred)
    if use_xgb: preds_list.append(xgb_pred)
    if use_lstm: preds_list.append(lstm_pred.flatten())

    hybrid_pred = np.mean(np.array(preds_list), axis=0)

    # ---------------------------
    # Evaluation
    # ---------------------------
    st.subheader("📌 Model Performance")
    def metrics(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SARIMA MAE", round(metrics(y_test[:len(sarima_pred)], sarima_pred)[0], 2))
    col2.metric("XGBoost MAE", round(metrics(y_test[:len(xgb_pred)], xgb_pred)[0], 2))
    col3.metric("LSTM MAE", round(metrics(y_test[:len(lstm_pred)], lstm_pred)[0], 2))
    col4.metric("Hybrid MAE", round(metrics(y_test[:len(hybrid_pred)], hybrid_pred)[0], 2))

    # ---------------------------
    # SHAP Explainable AI (XGBoost)
    # ---------------------------
    if use_xgb:
        st.subheader("🧠 SHAP Feature Importance")
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_train)
        fig_shap, ax = plt.subplots()
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        st.pyplot(fig_shap)

    # ---------------------------
    # Future Forecast
    # ---------------------------
    st.subheader("🔮 Future Forecast")
    future_steps = st.slider("Select months to predict", 1, 12, 6)

    # SARIMA
    if use_sarima:
        sarima_future = sarima_model.predict(n_periods=future_steps)
    else:
        sarima_future = np.repeat(df['Sales'].mean(), future_steps)

    # XGBoost
    if use_xgb:
        last_lags = df['Sales'].values[-6:].tolist()
        xgb_future = []
        for _ in range(future_steps):
            input_data = np.array(last_lags[-6:]).reshape(1, -1)
            pred = xgb_model.predict(input_data)[0]
            xgb_future.append(pred)
            last_lags.append(pred)
    else:
        xgb_future = np.repeat(df['Sales'].mean(), future_steps)

    # LSTM
    if use_lstm:
        lstm_input = scaled[-6:].reshape(1, 6, 1)
        lstm_future = []
        for _ in range(future_steps):
            pred = att_model.predict(lstm_input)
            lstm_future.append(pred[0, 0])
            lstm_input = np.append(lstm_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
        lstm_future = scaler.inverse_transform(np.array(lstm_future).reshape(-1, 1))
    else:
        lstm_future = np.repeat(df['Sales'].mean(), future_steps)

    # Hybrid
    preds_future = []
    if use_sarima: preds_future.append(sarima_future)
    if use_xgb: preds_future.append(xgb_future)
    if use_lstm: preds_future.append(lstm_future.flatten())

    hybrid_future = np.mean(np.array(preds_future), axis=0)
    future_dates = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=future_steps, freq='MS')
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Sales': hybrid_future})

    fig2 = px.line(pd.concat([df.reset_index(), future_df]), x='Date', y=['Sales', 'Predicted Sales'], title='Hybrid AI Future Forecast')
    st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------
    # Confidence Intervals
    # ---------------------------
    st.subheader("⚠️ Risk Analysis / Confidence Intervals")
    std_dev = np.std(hybrid_pred)
    upper = hybrid_future + 1.96 * std_dev
    lower = hybrid_future - 1.96 * std_dev
    risk_df = pd.DataFrame({'Date': future_dates, 'Lower Bound': lower, 'Prediction': hybrid_future, 'Upper Bound': upper})
    fig_ci = px.line(risk_df, x="Date", y=["Lower Bound", "Prediction", "Upper Bound"], title="Forecast with Confidence Intervals")
    st.plotly_chart(fig_ci, use_container_width=True)

    st.success("Forecast and analysis generated successfully!")
