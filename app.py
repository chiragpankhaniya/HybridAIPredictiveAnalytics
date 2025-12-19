# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Streamlit page config
st.set_page_config(page_title="Hybrid AI Predictive Analytics", layout="wide")
st.title("Hybrid AI-Driven Predictive Analytics")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload CSV with 'Date' and 'Sales' columns", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Check for required columns
    if 'Date' not in df.columns or 'Sales' not in df.columns:
        st.error("CSV must contain 'Date' and 'Sales' columns")
    else:
        st.success("File loaded successfully!")
        
        # Preprocess data
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True)
        
        # Fill missing 'Sales' values
        df['Sales'] = df['Sales'].fillna(df['Sales'].mean())  # or use interpolation
        
        st.write(df.head())

        y = df['Sales'].values

        # --- Train-Test Split (80-20) ---
        train_size = int(len(y) * 0.8)
        y_train, y_test = y[:train_size], y[train_size:]

        # --- SARIMA Model ---
        sarima_forecast = np.zeros(len(y_test))
        try:
            sarima_model = auto_arima(
                y_train, seasonal=True, m=12,
                stepwise=True, suppress_warnings=True
            )
            sarima_forecast = sarima_model.predict(n_periods=len(y_test))
        except Exception as e:
            st.warning(f"SARIMA failed: {e}")

        # --- XGBoost Model ---
        def create_features(y):
            df_feat = pd.DataFrame({'y': y})
            for i in range(1, 4):  # 3 lag features
                df_feat[f'lag_{i}'] = df_feat['y'].shift(i)
            df_feat = df_feat.dropna()
            X = df_feat.drop('y', axis=1).values
            y_target = df_feat['y'].values
            return X, y_target

        X_train, y_xgb_train = create_features(y_train)
        X_test, y_xgb_test = create_features(y_test)

        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50)
        xgb_model.fit(X_train, y_xgb_train)
        xgb_forecast = xgb_model.predict(X_test)

        # --- LSTM Model ---
        scaler = MinMaxScaler(feature_range=(0,1))
        y_scaled = scaler.fit_transform(y.reshape(-1,1))

        def create_lstm_dataset(dataset, look_back=3):
            X, Y = [], []
            for i in range(len(dataset)-look_back):
                X.append(dataset[i:(i+look_back), 0])
                Y.append(dataset[i + look_back, 0])
            return np.array(X), np.array(Y)

        look_back = 3
        X_lstm, y_lstm = create_lstm_dataset(y_scaled, look_back)
        train_size_lstm = int(len(X_lstm) * 0.8)
        X_train_lstm, X_test_lstm = X_lstm[:train_size_lstm], X_lstm[train_size_lstm:]
        y_train_lstm, y_test_lstm = y_lstm[:train_size_lstm], y_lstm[train_size_lstm:]

        X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
        X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

        lstm_model = Sequential()
        lstm_model.add(LSTM(20, input_shape=(look_back,1)))  # smaller LSTM for speed
        lstm_model.add(Dense(1))
        lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=1, verbose=0)

        lstm_forecast_scaled = lstm_model.predict(X_test_lstm)
        lstm_forecast = scaler.inverse_transform(lstm_forecast_scaled).flatten()

        # --- Ensemble Forecast ---
        min_len = min(len(y_test), len(sarima_forecast), len(xgb_forecast), len(lstm_forecast))
        ensemble_forecast = (sarima_forecast[-min_len:] + xgb_forecast[-min_len:] + lstm_forecast[-min_len:]) / 3

        # --- Display Results ---
        st.subheader("Predictions vs Actuals")
        fig, ax = plt.subplots(figsize=(10,5))
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
        last_data = y[-12:] if len(y) >= 12 else y  # last 12 months or all
        future_preds = []
        temp_series = last_data.copy()
        for i in range(6):
            try:
                sar_pred = auto_arima(temp_series, seasonal=True, m=12,
                                      stepwise=True, suppress_warnings=True).predict(n_periods=1)[0]
            except:
                sar_pred = temp_series[-1]
            X_temp, _ = create_features(temp_series)
            xgb_pred = xgb_model.predict(X_temp[-1].reshape(1,-1))[0]
            lstm_input = scaler.transform(temp_series[-look_back:].reshape(-1,1)).reshape(1, look_back, 1)
            lstm_pred = scaler.inverse_transform(lstm_model.predict(lstm_input)).flatten()[0]
            ensemble_next = (sar_pred + xgb_pred + lstm_pred) / 3
            future_preds.append(ensemble_next)
            temp_series = np.append(temp_series, ensemble_next)

        future_dates = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=6, freq='MS')
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Sales': future_preds})
        st.write(future_df)

        # Optional: Lazy SHAP Explainability
        if st.checkbox("Show SHAP Explainability for XGBoost"):
            import shap
            explainer = shap.Explainer(xgb_model)
            shap_values = explainer(X_test)
            st.pyplot(shap.summary_plot(shap_values, X_test, show=False))

        st.success("Prediction Completed!")

else:
    st.info("Upload a CSV file to start predictions.")
