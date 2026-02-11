import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import streamlit as st
from config import NIFTY_50_STOCKS, HISTORICAL_DATA_START_DATE

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker: str, start_date: str = HISTORICAL_DATA_START_DATE) -> pd.DataFrame:
    """
    Fetch historical stock data for a given ticker and start date.
    
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for fetching data in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: DataFrame containing historical stock data.
    """
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data.reset_index(inplace=True)

        if stock_data.empty:
            st.warning(f"No data found for {ticker}.")
            return pd.DataFrame()
        stock_data['returns'] = stock_data['Close'].pct_change().dropna()
        
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
    
@st.cache_data(ttl=3600)
def fetch_nifty_50_data(start_date: str = HISTORICAL_DATA_START_DATE) -> pd.DataFrame:
    """
    Fetch historical stock data for Nifty 50 index.
    
    Returns:
        pd.DataFrame: Historical data DataFrames.
    """
    return fetch_stock_data('^NSEI', start_date)

@st.cache_data(ttl=3600)
def fetch_multiple_stocks(tickers: List[str], start_date: str = HISTORICAL_DATA_START_DATE) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical stock data for multiple tickers.
    
    Args:
        tickers (List[str]): List of stock ticker symbols.
        start_date (str): Start date for fetching data in 'YYYY-MM-DD' format.
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping ticker symbols to their historical data DataFrames.
    """
    stock_data_dict = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx,ticker in enumerate(tickers):
        status_text.text(f"Fetching data for {ticker}...")
        df = fetch_stock_data(ticker, start_date)
        if not df.empty:
            stock_data_dict[ticker] = df
        progress_bar.progress((idx + 1) / len(tickers))
    status_text.text("Data fetching complete.")
    progress_bar.empty()
    status_text.empty()
    return stock_data_dict

def get_stock_sector(ticker: str) -> str:
    """
    Get a mapping of stock tickers to their respective sectors.
    
    Returns:
        Dict[str, str]: Dictionary mapping stock tickers to sectors.
    """
    return NIFTY_50_STOCKS.get(ticker, "Unknown Sector")


