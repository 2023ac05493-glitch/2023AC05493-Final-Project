import yfinance as yf
import pandas as pd
# Fetch Nifty-50 and VIX daily
nifty_data = yf.download('^NSEI', start='2020-01-01', end='2026-01-26')
vix_data = yf.download('^INDIAVIX', start='2020-01-01', end='2026-01-26')

# Combine into single DataFrame
df = pd.DataFrame({
    'nifty_close': nifty_data['Close'],
    'nifty_high': nifty_data['High'],
    'nifty_low': nifty_data['Low'],
    'nifty_volume': nifty_data['Volume'],
    'india_vix': vix_data['Close']
})
