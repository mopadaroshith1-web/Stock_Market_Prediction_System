
# Stock Market Prediction System

## Project Overview
A machine learning application that analyzes historical stock market data and estimates future closing prices.

## Features
- Historical stock price visualization
- Trading volume analysis
- 20-day and 50-day moving averages
- Exploratory Data Analysis (EDA)
- Linear Regression and Random Forest models
- Model evaluation using MAE, MSE, RMSE and R²
- Next-trading-day price estimation
- Downloadable CSV files
- Interactive Streamlit dashboard

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- yfinance
- Streamlit

## Installation

Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run app.py
```

## Data Source
Historical market data is retrieved using Yahoo Finance through the yfinance library.

## Disclaimer
This project is for educational purposes only. Predictions are estimates and are not financial advice.
