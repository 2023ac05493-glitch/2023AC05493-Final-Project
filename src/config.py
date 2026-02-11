import os
from dotenv import load_dotenv

load_dotenv()

NIFTY_50_STOCKS={
    'ADANIENT.NS':'Metals & Mining',
    'ADANIPORTS.NS':'Services',
    'APOLLOHOSP.NS':'Healthcare',
    'ASIANPAINT.NS':'Consumer Goods',
    'AXISBANK.NS':'Financial Services',
    'BAJAJ-AUTO.NS':'Automobile',
    'BAJFINANCE.NS':'Financial Services',
    'BAJAJFINSV.NS':'Financial Services',
    'BPCL.NS':'Oil & Gas',
    'BHARTIARTL.NS':'Telecom',
    'BRITANNIA.NS':'Consumer Goods',
    'CIPLA.NS':'Pharmaceuticals',
    'COALINDIA.NS':'Metals & Mining',
    'DRREDDY.NS':'Pharmaceuticals',
    'EICHERMOT.NS':'Automobile',
    'GRASIM.NS':'Cement & Cement Products',
    'HCLTECH.NS':'IT',
    'HDFCBANK.NS':'Financial Services',
    'HDFCLIFE.NS':'Financial Services',
    'HEROMOTOCO.NS':'Automobile',
    'HINDALCO.NS':'Metals & Mining',
    'HINDUNILVR.NS':'Consumer Goods',
    'ICICIBANK.NS':'Financial Services',
    'ITC.NS':'Consumer Goods',
    'INFY.NS':'IT',
    'INDUSINDBK.NS':'Financial Services',
    'JSWSTEEL.NS':'Metals & Mining',
    'KOTAKBANK.NS':'Financial Services',
    'LT.NS':'Automobile',
    'M&M.NS':'Automobile',
    'MARUTI.NS':'Automobile',
    'NESTLEIND.NS':'Consumer Goods',
    'NTPC.NS':'Power',
    'ONGC.NS':'Oil & Gas',
    'POWERGRID.NS':'Power',
    'RELIANCE.NS':'Oil & Gas',
    'SBIN.NS':'Financial Services',
    'SBILIFE.NS':'Financial Services',
    'SHRIRAMFIN.NS':'Financial Services',
    'SUNPHARMA.NS':'Pharmaceuticals',
    'TCS.NS':'IT',
    'TATASTEEL.NS':'Metals & Mining',
    'TATACONSUM.NS':'Consumer Goods',
    'TECHM.NS':'IT',
    'TITAN.NS':'Consumer Goods',
    'TRENT.NS':'Consumer Goods',
    'ULTRACEMCO.NS':'Cement & Cement Products',
    'WIPRO.NS':'IT'
}

VAR_CONFIDENCE_LEVELS = [0.95, 0.99]
VAR_PREDICTION_DAYS = 7
HISTORICAL_DATA_START_DATE = '2020-01-01'

GARCH_P=1
GARCH_Q=1

NEWS_API_KEY = os.getenv('NEWS_API_KEY','')

HF_MODEL_NAME = os.getenv('HF_MODEL_NAME','google/flan-t5-base')

APP_TITLE = "VaR Prediction Workstation"
APP_ICON = "📊"