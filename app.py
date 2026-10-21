
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

st.set_page_config(
    page_title="Stock Market Prediction System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Market Prediction System")
st.write(
    "Analyze historical stock prices and predict the next "
    "trading day's closing price using Machine Learning."
)

st.sidebar.header("Stock Settings")

stocks = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "Reliance Industries": "RELIANCE.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS"
}

company = st.sidebar.selectbox(
    "Choose a company",
    list(stocks.keys())
)

period = st.sidebar.selectbox(
    "Select historical period",
    ["1y", "2y", "5y", "10y"],
    index=1
)

ticker = stocks[company]

st.sidebar.info(f"Stock symbol: {ticker}")

@st.cache_data(ttl=300)
def load_data(symbol, selected_period):
    data = yf.download(
        symbol,
        period=selected_period,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    # Handle newer yfinance versions that return MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data.columns = [
        str(column).lower().replace(" ", "_")
        for column in data.columns
    ]

    required = ["date", "open", "high", "low", "close", "volume"]
    data = data[required]

    data["date"] = pd.to_datetime(data["date"])
    data = data.drop_duplicates(subset=["date"])
    data = data.dropna()
    data = data.sort_values("date").reset_index(drop=True)

    return data

with st.spinner("Downloading historical stock data..."):
    try:
        df = load_data(ticker, period)
    except Exception as error:
        st.error(f"Unable to download stock data: {error}")
        st.stop()

if df.empty or len(df) < 30:
    st.error(
        "Not enough stock data was downloaded. "
        "Check your internet connection or choose another company."
    )
    st.stop()

# Feature engineering
df["MA20"] = df["close"].rolling(20).mean()
df["MA50"] = df["close"].rolling(50).mean()
df["daily_return"] = df["close"].pct_change() * 100

# Dashboard metrics
latest_close = float(df["close"].iloc[-1])
previous_close = float(df["close"].iloc[-2])
change = latest_close - previous_close
change_pct = (change / previous_close) * 100

st.subheader(f"{company} ({ticker})")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Latest Closing Price",
    f"{latest_close:,.2f}",
    f"{change_pct:+.2f}%"
)

col2.metric(
    "Highest Price",
    f"{df['high'].max():,.2f}"
)

col3.metric(
    "Lowest Price",
    f"{df['low'].min():,.2f}"
)

col4.metric(
    "Trading Volume",
    f"{int(df['volume'].iloc[-1]):,}"
)

st.caption(
    "Prices are shown in the stock's quoted currency. "
    "Market data may be delayed or unavailable."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Market Overview",
    "🔍 Data Analysis",
    "🤖 Price Prediction",
    "📋 Raw Data"
])

with tab1:
    st.subheader("Historical Closing Price")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["close"], label="Closing Price")
    ax.plot(df["date"], df["MA20"], label="20-Day Moving Average")
    ax.plot(df["date"], df["MA50"], label="50-Day Moving Average")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"{company} Stock Price Trend")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Trading Volume")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(df["date"], df["volume"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.set_title("Historical Trading Volume")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Descriptive Statistics**")
        st.dataframe(
            df[["open", "high", "low", "close", "volume"]]
            .describe()
            .round(2),
            use_container_width=True
        )

    with col2:
        st.write("**Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            df[["open", "high", "low", "close", "volume"]]
            .corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax
        )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Daily Percentage Returns")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], df["daily_return"])
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    ax.set_title("Daily Stock Returns")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write(
        f"**Average closing price:** {df['close'].mean():,.2f}"
    )
    st.write(
        f"**Price standard deviation:** {df['close'].std():,.2f}"
    )
    st.write(
        f"**Records after cleaning:** {len(df)}"
    )

with tab3:
    st.subheader("Machine Learning Stock Price Prediction")

    st.write(
        "The models use historical price and trading-volume features "
        "to estimate the next trading day's closing price."
    )

    # Build next-day prediction target without future leakage
    model_df = df.copy()
    model_df["previous_close"] = model_df["close"].shift(1)
    model_df["previous_2_close"] = model_df["close"].shift(2)
    model_df["previous_3_close"] = model_df["close"].shift(3)
    model_df["previous_volume"] = model_df["volume"].shift(1)
    model_df["target"] = model_df["close"].shift(-1)

    features = [
        "close",
        "previous_close",
        "previous_2_close",
        "previous_3_close",
        "volume",
        "previous_volume",
        "MA20",
        "MA50"
    ]

    model_df = model_df.dropna().reset_index(drop=True)

    if len(model_df) < 30:
        st.warning("Not enough records to train the models.")
    else:
        X = model_df[features]
        y = model_df["target"]

        split = int(len(model_df) * 0.8)

        X_train = X.iloc[:split]
        X_test = X.iloc[split:]
        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

        linear_model = LinearRegression()
        forest_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            min_samples_leaf=2
        )

        linear_model.fit(X_train, y_train)
        forest_model.fit(X_train, y_train)

        predictions_linear = linear_model.predict(X_test)
        predictions_forest = forest_model.predict(X_test)

        def calculate_metrics(actual, predicted):
            return {
                "MAE": mean_absolute_error(actual, predicted),
                "MSE": mean_squared_error(actual, predicted),
                "RMSE": np.sqrt(
                    mean_squared_error(actual, predicted)
                ),
                "R2 Score": r2_score(actual, predicted)
            }

        linear_metrics = calculate_metrics(
            y_test, predictions_linear
        )
        forest_metrics = calculate_metrics(
            y_test, predictions_forest
        )

        metrics_table = pd.DataFrame(
            [linear_metrics, forest_metrics],
            index=["Linear Regression", "Random Forest"]
        )

        st.write("**Model Performance on Test Data**")
        st.dataframe(
            metrics_table.round(4),
            use_container_width=True
        )

        # Select the model with the lower test MAE
        if linear_metrics["MAE"] <= forest_metrics["MAE"]:
            best_model = linear_model
            best_name = "Linear Regression"
        else:
            best_model = forest_model
            best_name = "Random Forest"

        st.success(f"Best model based on test MAE: {best_name}")

        st.subheader("Actual vs Predicted Closing Prices")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            model_df["date"].iloc[split:],
            y_test.values,
            label="Actual Closing Price"
        )
        ax.plot(
            model_df["date"].iloc[split:],
            predictions_linear,
            label="Linear Regression Prediction"
        )
        ax.plot(
            model_df["date"].iloc[split:],
            predictions_forest,
            label="Random Forest Prediction"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Next Trading Day Estimate")

        # Refit on all available labeled data before forecasting
        best_model.fit(X, y)

        latest = df.iloc[-1]
        previous = df.iloc[-2]
        previous_2 = df.iloc[-3]
        previous_3 = df.iloc[-4]

        next_features = pd.DataFrame([{
            "close": latest["close"],
            "previous_close": previous["close"],
            "previous_2_close": previous_2["close"],
            "previous_3_close": previous_3["close"],
            "volume": latest["volume"],
            "previous_volume": previous["volume"],
            "MA20": df["MA20"].iloc[-1],
            "MA50": df["MA50"].iloc[-1]
        }])[features]

        next_price = float(best_model.predict(next_features)[0])
        estimated_change = next_price - latest_close
        estimated_pct = (estimated_change / latest_close) * 100

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Latest Actual Close",
            f"{latest_close:,.2f}"
        )
        col2.metric(
            "Estimated Next Close",
            f"{next_price:,.2f}",
            f"{estimated_change:+,.2f}"
        )
        col3.metric(
            "Estimated Change",
            f"{estimated_pct:+.2f}%"
        )

        st.warning(
            "This is an educational machine learning estimate, "
            "not a guaranteed price or financial advice. "
            "The next trading session's date may vary due to "
            "market holidays."
        )

        results = pd.DataFrame({
            "Date": model_df["date"].iloc[split:].values,
            "Actual Price": y_test.values,
            "Linear Regression Prediction": predictions_linear,
            "Random Forest Prediction": predictions_forest
        })

        st.download_button(
            "Download Test Predictions CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_predictions.csv",
            mime="text/csv"
        )

with tab4:
    st.subheader("Historical Stock Dataset")

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download Historical Data CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_historical_data.csv",
        mime="text/csv"
    )

st.divider()

st.caption(
    "Stock Market Prediction System | Built with Python, "
    "Pandas, Scikit-learn, Yahoo Finance and Streamlit"
)
