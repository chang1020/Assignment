import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

# List of Malaysian banks and their tickers
malaysian_banks = {
    "Alliance Bank Malaysia Berhad": "2488.KL",
    "Affin Bank Berhad": "5185.KL",
    "AMMB Holdings Berhad": "1015.KL",
    "BIMB Holdings Berhad": "5258.KL",
    "CIMB Group Holdings Berhad": "1023.KL",
    "Hong Leong Bank Berhad": "5819.KL",
    "Hong Leong Financial Group Berhad": "1082.KL",
    "Malayan Banking Berhad": "1155.KL",
    "Public Bank Berhad": "1295.KL",
    "RHB Bank Berhad": "1066.KL"
}

def load_data(ticker):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=1825)).strftime('%Y-%m-%d')  # 5 years of data
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def train_model(data, model):
    data['next_close'] = data['Close'].shift(-1)  # Ensure correct column name
    data.dropna(inplace=True)
    X = data[['Open', 'High', 'Low', 'Close', 'Volume']]  # Ensure correct column names
    y = data['next_close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, mse

def predict_next_day(model, last_row):
    next_day = model.predict([last_row[['Open', 'High', 'Low', 'Close', 'Volume']]])  # Ensure correct column names
    return next_day[0]

def plot_prices(data, gbr_model, rf_model):
    data['gbr_predicted_next_close'] = gbr_model.predict(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    data['rf_predicted_next_close'] = rf_model.predict(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=data.index, y=data['gbr_predicted_next_close'], mode='lines', name='GBR Predicted'))
    fig.add_trace(go.Scatter(x=data.index, y=data['rf_predicted_next_close'], mode='lines', name='RF Predicted'))
    
    fig.update_layout(
        title='Actual vs Predicted Prices',
        xaxis_title='Date',
        yaxis_title='Close Price',
        xaxis_rangeslider_visible=True,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="minute", stepmode="backward"),
                    dict(count=30, label="30m", step="minute", stepmode="backward"),
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=1, label="1mo", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
    )
    
    return fig

def page2(conn, ticker):
    st.title("Malaysian Bank Stock Price Prediction")
    st.write("""
    ## Program Description
    This program retrieves historical stock data for selected Malaysian banks and predicts the next day's closing price using machine learning models.
    """)

    # Dropdown for selecting a Malaysian bank or "None"
    selected_bank = st.selectbox("Select a Malaysian Bank:", ["None"] + list(malaysian_banks.keys()), format_func=lambda x: f"{x} ({malaysian_banks.get(x, '')})", key="selectbox2")
    
    # Input box for entering a ticker symbol directly
    ticker_input = st.text_input("Or enter Ticker Symbol directly (e.g., 1155.KL)", key="ticker_input2")

    # Determine the ticker based on selection or input
    if ticker_input:
        ticker = ticker_input
    elif selected_bank != "None":
        ticker = malaysian_banks[selected_bank]

    if st.button("Predict"):
        if ticker is None:
            st.error("Please select a Malaysian bank or enter a ticker symbol.")
        else:
            # Retrieve and load data
            data = load_data(ticker)
            if not data.empty:
                st.write(data.tail())

                # Train Gradient Boosting Regressor
                gbr_model, gbr_mse = train_model(data, GradientBoostingRegressor())
                gbr_next_price = predict_next_day(gbr_model, data.iloc[-1])

                # Train Random Forest Regressor
                rf_model, rf_mse = train_model(data, RandomForestRegressor())
                rf_next_price = predict_next_day(rf_model, data.iloc[-1])

                # Display predictions and MSE in a table
                st.write("### Predictions and Mean Squared Error (MSE)")
                st.table(pd.DataFrame({
                    "Model": ["Gradient Boosting Regressor", "Random Forest Regressor"],
                    "Predicted Next Close Price": [gbr_next_price, rf_next_price],
                    "Mean Squared Error (MSE)": [gbr_mse, rf_mse]
                }))

                # Visualization of actual vs predicted prices
                st.write('### Actual vs Predicted Prices')
                fig = plot_prices(data, gbr_model, rf_model)
                st.plotly_chart(fig)

            else:
                st.error(f"No data found for ticker symbol: {ticker}")

if __name__ == "__main__":
    # Dummy connection object for testing purpose
    conn = None
    page2(conn, None)
