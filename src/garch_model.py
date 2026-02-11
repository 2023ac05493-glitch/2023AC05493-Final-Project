import numpy as np 
import pandas as pd
from arch import arch_model
from scipy import stats
from typing import Tuple, Dict
import streamlit as st
from config import GARCH_P, GARCH_Q, VAR_CONFIDENCE_LEVELS, VAR_PREDICTION_DAYS

class GARCHVaRModel:
    def __init__(self, returns: pd.Series, p: int = GARCH_P, q: int = GARCH_Q):
        self.returns = returns*100
        self.p = p
        self.q = q
        self.model = None
        self.fitted_model = None
        self.forecasts = None

    def fit(self):
        try:
            self.model = arch_model(self.returns, vol='Garch', p=self.p, q=self.q, dist='normal')

            self.fitted_model = self.model.fit(disp='off',show_warning=False)

            return True
        except Exception as e:
            st.error(f"Error fitting GARCH model: {e}")
            return False

    def forecast_volatility(self, horizon: int = VAR_PREDICTION_DAYS) -> pd.DataFrame:
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting.")
        self.forecasts = self.fitted_model.forecast(horizon=horizon, reindex=False)

        variance_forecast = self.forecasts.variance.values[-1, :]

        forecast_df = pd.DataFrame({
            'Day': np.arange(1, horizon + 1),
            'Variance': variance_forecast,
            'Volatility': np.sqrt(variance_forecast)
        })

        return forecast_df

    def calculate_var(self, confidence_level: float, horizon: int = VAR_PREDICTION_DAYS) -> Dict:
        forecast_df = self.forecast_volatility(horizon)

        z_score = stats.norm.ppf(1 - confidence_level)

        cumulative_variance = forecast_df['Variance'].sum()
        cumulative_volatility = np.sqrt(cumulative_variance)

        var_percentage = z_score * cumulative_volatility

        daily_vars=[]
        for day in range(1, horizon + 1):
            daily_variance = forecast_df.loc[forecast_df['Day'] <= day, 'Variance'].sum()
            daily_volatility = np.sqrt(daily_variance)
            daily_var = z_score * daily_volatility
            daily_vars.append(daily_var)
        return {
            'confidence_level': confidence_level,
            'horizon': horizon,
            'var_percentage': var_percentage,
            'cumulative_volatility': cumulative_volatility,
            'daily_vars': daily_vars,
            'forecast_df': forecast_df
        }
    
    def get_model_summary(self) -> str:
        if self.fitted_model is None:
            raise ValueError("Model must be fitted to get summary.")
        return str(self.fitted_model.summary())

def rolling_var_backtest(returns: pd.Series, window: int =252, horizon: int = VAR_PREDICTION_DAYS, confidence_level: float = 0.05) -> pd.DataFrame:
    results=   []

    if len(returns) < window + horizon:
        st.warning("Not enough data for backtesting. Increase the window size or reduce the horizon.")
        return pd.DataFrame()
    for i in range(window,len(returns) - horizon, horizon):
        train_returns = returns.iloc[i - window:i]
        test_returns = returns.iloc[i:i + horizon]

        model = GARCHVaRModel(train_returns)
        if model.fit():

            var_result = model.calculate_var(confidence_level, horizon)

            actual_return = (test_returns*100).sum()

            results.append({
                'date': returns.index[i],
                'predicted_var': var_result['var_percentage'],
                'actual_return': actual_return,
                'var_breach': actual_return < var_result['var_percentage']
            })
    return pd.DataFrame(results)

def calculate_var_for_multiple_stocks(stock_dict: Dict[str, pd.DataFrame], confidence_level: list = VAR_CONFIDENCE_LEVELS, horizon: int = VAR_PREDICTION_DAYS) -> pd.DataFrame:
    results = []
    for ticker, df in stock_dict.items():
        if 'returns' not in df.columns:
            st.warning(f"Returns not calculated for {ticker}. Skipping.")
            continue
        returns = df['returns'].dropna()

        model = GARCHVaRModel(returns)
        if model.fit():
            for confidence in confidence_level:
                var_result = model.calculate_var(confidence, horizon)
                results.append({
                    'ticker': ticker,
                    'confidence_level': f"{confidence*100:.2f}%",
                    'var_percentage': var_result['var_percentage'],
                    'day1_var': var_result['daily_vars'][0],
                    'volatility': var_result['cumulative_volatility']
                })
    return pd.DataFrame(results)
        