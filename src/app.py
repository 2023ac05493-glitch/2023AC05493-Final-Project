import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict,List

from config import (NIFTY_50_STOCKS, APP_ICON, APP_TITLE)

from market_data_loader import(
    fetch_stock_data, fetch_nifty_50_data, fetch_multiple_stocks,
    get_stock_sector
)

from garch_model import (
    GARCHVaRModel, rolling_var_backtest, calculate_var_for_multiple_stocks
)

from news_agent import get_news_agent

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .metric-card {
            background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
            font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
        color: #333;
    }
    .metric-label {
            font-size: 14px;
            opacity: 0.9;
    }
    .stButton>button {
            width: 100%;
        background-color: #667eea;
        color: white;
        border-radius: 5px;
        padding: 10px;
        font-weight: bold;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
</style>                
""", unsafe_allow_html=True)

def display_var_card(title: str, var_value: float, confidence: str, color: str = "#667eea", delta: float = None):
    delta_str = f" ({delta:+.2f}%)" if delta is not None else ""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color} 0%, #764ba2 100%);
                padding: 20px; border-radius: 10px; text-align: center; color: white;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <div style="font-size: 14px; opacity: 0.9;">{title}</div>
        <div style="font-size: 32px; font-weight: bold; margin: 10px 0;">{var_value:.2f}%</div>
        <div style="font-size: 12px; opacity: 0.7;">Confidence: {confidence}</div>
    </div>
    """, unsafe_allow_html=True)

def plot_true_vs_predicted_var(backtest_results: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=backtest_results['date'], y=backtest_results['actual_return'], mode='lines+markers', name='Actual Return',line=dict(color='blue', width=2),marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=backtest_results['date'], y=backtest_results['predicted_var'], mode='lines+markers', name='Predicted VaR',line=dict(color='red', width=2),marker=dict(size=6)))
    fig.add_hline(y=0, line_dash='dash',line_color='gray',opacity=0.5)
    fig.update_layout(title=f"Actual Returns vs Predicted VaR", xaxis_title="Date", yaxis_title="Returns (%)",hovermode='x unified',template='plotly_white',height=400)
    return fig

def plot_individual_stock_var(stock_var_df: pd.DataFrame, topn: int = 15):
    pivot_data = stock_var_df.pivot(index='ticker', columns='confidence_level', values='var_percentage').reset_index()

    if '95.00%' in pivot_data.columns:
        pivot_data = pivot_data.nlargest(topn, '95.00%')
    
    fig = go.Figure()

    if '95.00%' in pivot_data.columns:
        fig.add_trace(go.Bar(x=pivot_data['ticker'], y=pivot_data['95.00%'], name='VaR at 95%', marker_color='indianred'))
    if '99.00%' in pivot_data.columns: 
        fig.add_trace(go.Bar(x=pivot_data['ticker'], y=pivot_data['99.00%'], name='VaR at 99%', marker_color='lightsalmon'))

    fig.update_layout(title=f"VaR Predictions for Top {topn} Stocks", xaxis_title="Stock Ticker", yaxis_title="VaR (%)", barmode='group',template='plotly_white',height=400,xaxis_tickangle=-45)

    return fig

def main():
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown("Welcome to the VaR Prediction Workstation! This application allows you to analyze and predict the Value at Risk (VaR) for Nifty 50 stocks using GARCH models. Explore the historical data, backtest the model's performance, and visualize the results with interactive charts.")

    st.sidebar.header("Configuration")

    analysis_type = st.sidebar.radio("Select Analysis Type", ["Nifty 50 Index","Single Stock Analysis", "Multiple Stocks Analysis"])

    selected_stocks = []

    if analysis_type == "Single Stock Analysis":
        selected_stock = st.sidebar.selectbox("Select a stock for analysis", options = list(NIFTY_50_STOCKS.keys()), format_func=lambda x: f"{x.replace('.NS', '')} - {NIFTY_50_STOCKS[x]}")
        selected_stocks = [selected_stock]  
    elif analysis_type == "Multiple Stocks Analysis":
        #include a select all option
        select_all = st.sidebar.checkbox("Select All Stocks")
        all_stocks = list(NIFTY_50_STOCKS.keys())
        if select_all:
            selected_stocks = all_stocks
        else:
            selected_stocks = st.sidebar.multiselect("Select stocks for analysis", options = all_stocks, format_func=lambda x: f"{x.replace('.NS', '')} - {NIFTY_50_STOCKS[x]}")

        selected_stocks = selected_stocks[:50]
    
    run_analysis = st.sidebar.button("Run Analysis", type="primary")

    if run_analysis or 'var_results' in st.session_state:
        if run_analysis:
            with st.spinner("Running analysis..."):
                if analysis_type == "Nifty 50 Index":
                    nifty_data = fetch_nifty_50_data()
                    if not nifty_data.empty:
                        st.session_state['single_data'] = nifty_data
                        st.session_state['ticker'] = '^NSEI'
                        st.session_state['analysis_type'] = 'single'
                elif analysis_type == "Single Stock Analysis" and selected_stocks:
                    stock_data = fetch_stock_data(selected_stocks[0])
                    if not stock_data.empty:
                        st.session_state['single_data'] = stock_data
                        st.session_state['ticker'] = selected_stocks[0]
                        st.session_state['analysis_type'] = 'single'
                elif analysis_type == "Multiple Stocks Analysis" and selected_stocks:
                    stock_data_dict = fetch_multiple_stocks(selected_stocks)
                    if stock_data_dict:
                        st.session_state['multi_data'] = stock_data_dict
                        st.session_state['analysis_type'] = 'multiple'

        if st.session_state['analysis_type'] == 'single':
            display_single_stock_analysis()
        elif st.session_state['analysis_type'] == 'multiple':
            display_multiple_stocks_analysis()
    else:
        st.info("Please select an analysis type and click 'Run Analysis' to see the results.")


        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            ### GARCH Model Overview
            - GARCH(1,1) volatility forecasting
            - 7 day VaR prediction
            - 95% and 99% confidence levels
            """)
        with col2:
            st.markdown("""
            ### Application Features
            - Historical data visualization
            - Backtesting with rolling window
            - Interactive charts for VaR analysis
            """)
        with col3:
            st.markdown("""
            ### AI Assistant
            - Context-aware responses
            - Explain model outputs
            - Risk insights and recommendations
            """)
    st.markdown("---")
    st.header("AI Assistant")
    display_chat_interface()
def display_single_stock_analysis():
    data = st.session_state['single_data']
    ticker = st.session_state['ticker']

    st.subheader(f"VaR analysis for {ticker.replace('.NS', '')}")
    st.dataframe(data.tail(10))
    st.markdown("---")
    returns = data['returns'].dropna()
    model = GARCHVaRModel(returns)
    if model.fit():
        var_result_95 = model.calculate_var(0.95)
        var_result_99 = model.calculate_var(0.99)
        
        st.subheader("VaR Predictions")
        col1, col2 = st.columns(2)

        with col1:
            display_var_card(f"Next day VAR - {ticker}", var_result_95['daily_vars'][0], "95% Confidence", color="#3498db")
        with col2:
            display_var_card(f"Next day VaR - {ticker}", var_result_99['daily_vars'][0], "99% Confidence", color="#e74c3c")
        
        st.markdown("---")
        st.markdown("### 7-Day Cumulative VaR Predictions")
        col3,col4 = st.columns(2)

        with col3:
            display_var_card(f"7 day VAR - {ticker}", var_result_95['var_percentage'], "95% Confidence", color="#3498db")
        with col4:
            display_var_card(f"7 day VaR - {ticker}", var_result_99['var_percentage'], "99% Confidence", color="#e74c3c")
        st.markdown("---")
        st.subheader("VaR Progression Over 7 Days")

        daily_var_df = pd.DataFrame({
            'Day': list(range(1, 8)),
            'VaR at 95%': var_result_95['daily_vars'],
            'VaR at 99%': var_result_99['daily_vars']
        })

        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter
            (x=daily_var_df['Day'], y=daily_var_df['VaR at 95%'], mode='lines+markers', name='VaR at 95%',line=dict(color='blue', width=2),marker=dict(size=6)))
        fig_daily.add_trace(go.Scatter
            (x=daily_var_df['Day'], y=daily_var_df['VaR at 99%'], mode='lines+markers', name='VaR at 99%',line=dict(color='red', width=2),marker=dict(size=6)))
        fig_daily.update_layout(xaxis_title="Day", yaxis_title="VaR (%)", title=f"VaR Progression for {ticker.replace('.NS', '')}", template='plotly_white', height=400)
        
        st.plotly_chart(fig_daily, use_container_width=True)
        st.markdown("---")
        st.subheader("True vs Predicted VaR Backtest")
        with st.spinner("Running backtest..."):
            backtest_95 = rolling_var_backtest(returns, confidence_level=0.95)

            if not backtest_95.empty:
                fig_backtest = plot_true_vs_predicted_var(backtest_95)
                st.plotly_chart(fig_backtest, use_container_width=True) 

                breach_rate_95 = backtest_95['var_breach'].sum() / len(backtest_95) * 100
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Prediction", len(backtest_95))
                with col2:
                    st.metric("VaR Breaches", backtest_95['var_breach'].sum())
                with col3:
                    st.metric("Breach Rate", f"{breach_rate_95:.2f}%")
        st.session_state['var_context'] = f"""
        Current VaR Analysis for {ticker}:
        - Next-day VaR (95%): {var_result_95['daily_vars'][0]}
        - Next-day VaR (99%): {var_result_99['daily_vars'][0]}
        - 7-Day VaR (95%): {var_result_95['var_percentage']}
        - 7-Day VaR (99%): {var_result_99['var_percentage']}
        - Volatility : {var_result_95['cumulative_volatility']}
        """
def plot_sector_var_breakdown(var_results: pd.DataFrame, confidence_level: str = '95.00%'):
    filtered_results = var_results[var_results['confidence_level'] == confidence_level].copy()
    filtered_results['sector'] = filtered_results['ticker'].apply(get_stock_sector)

    sector_var = filtered_results.groupby('sector')['var_percentage'].agg(['mean', 'count']).reset_index()
    sector_var.columns = ['sector', 'average_var', 'stock_count']
    sector_var = sector_var.sort_values('average_var', ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(x=sector_var['average_var'], y=sector_var['sector'], 
    orientation='h', marker=dict(color=sector_var['average_var'], 
    colorscale='Viridis', showscale=True,colorbar=dict(title='Avg VaR (%)')),
                         hovertemplate='<b>%{y}</b><br>Avg VaR: %{x:.2f}%<br>Stocks: %{customdata[0]}<extra></extra>',
                         customdata=sector_var[['stock_count']]))
    fig.update_layout(title=f"Average VaR by Sector at {confidence_level}", xaxis_title="Average VaR (%)", yaxis_title="Sector", template='plotly_white', height=500)
    return fig

def display_multiple_stocks_analysis():
    stock_dict = st.session_state['multi_data']
    with st.spinner("Calculating VaR for selected stocks..."):
        var_results = calculate_var_for_multiple_stocks(stock_dict)
        st.session_state['var_results'] = var_results.dropna()

    if not var_results.empty:
        st.subheader("VaR Predictions for Selected Stocks")
        
        var_95_data = var_results[var_results['confidence_level'] == '95.00%']
        var_99_data = var_results[var_results['confidence_level'] == '99.00%']

        #Next-day VaR cards
        st.markdown("### Next-day VaR Predictions")
        col1, col2 = st.columns(2)
        with col1:
            avg_next_day_var_95 = var_95_data['day1_var'].mean()
            display_var_card("Avg Next-day VaR at 95%", avg_next_day_var_95, confidence="95%", color="#3498db")
        with col2:
            avg_next_day_var_99 = var_99_data['day1_var'].mean()
            display_var_card("Avg Next-day VaR at 99%", avg_next_day_var_99, confidence="99%", color="#e74c3c")
        st.markdown("---")
        st.markdown("### 7-Day Cumulative VaR Predictions")
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            avg_var_95 = var_95_data['var_percentage'].mean()
            display_var_card("Avg VaR at 95%", avg_var_95,confidence="95%", color="#3498db")
        with col2:
            avg_var_99 = var_99_data['var_percentage'].mean()
            display_var_card("Avg VaR at 99%", avg_var_99,confidence="99%", color="#e74c3c")
        with col3:
            max_var_95 = var_95_data['var_percentage'].min()
            display_var_card("Max VaR at 95%", max_var_95, confidence="95%", color="#c0392b")
        with col4:
            min_var_95 = var_95_data['var_percentage'].max()
            display_var_card("Min VaR at 95%", min_var_95, confidence="95%", color="#27ae60")
        st.markdown("---")
        st.subheader("Individual Stock VaR Comparison")
        fig_stocks = plot_individual_stock_var(var_results)
        st.plotly_chart(fig_stocks, use_container_width=True)
        st.markdown("---")
        st.subheader("VaR by Sector")

        col1, col2 = st.columns(2)

        with col1:
            fig_sector_95 = plot_sector_var_breakdown(var_results, confidence_level='95.00%')
            st.plotly_chart(fig_sector_95, use_container_width=True)
        with col2:
            fig_sector_99 = plot_sector_var_breakdown(var_results, confidence_level='99.00%')
            st.plotly_chart(fig_sector_99, use_container_width=True)
        st.markdown("---")
        with st.expander("Detailed VaR Results"):
            display_df = var_results.copy()
            display_df['sector'] = display_df['ticker'].apply(get_stock_sector)
            display_df['var_percentage'] = display_df['var_percentage'].apply(lambda x: f"{abs(x):.2f}%")
            display_df['volatility'] = display_df['volatility'].apply(lambda x: f"{x:.2f}%")
            st.dataframe(display_df, use_container_width=True)
        st.session_state['var_context'] = f"""
        Portfolio VaR Analysis for {len(var_results)} stocks:
        - Average next-day VaR at 95%: {var_95_data['day1_var'].mean():.2f}%
        - Average next-day VaR at 99%: {var_99_data['day1_var'].mean():.2f}%
        - Average 7-Day VaR (95%): {var_95_data['var_percentage'].mean():.2f}%
        - Average 7-Day VaR (99%): {var_99_data['var_percentage'].mean():.2f}%
        - Minimum risk stock at 95%: {var_95_data.loc[var_95_data['var_percentage'].idxmax(), 'ticker']}
        - Maximum risk stock at 95%: {var_95_data.loc[var_95_data['var_percentage'].idxmin(), 'ticker']}
        """   

def display_chat_interface():
    st.subheader("AI Assistant")
    
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'news_loaded' not in st.session_state:
        st.session_state['news_loaded'] = False
    
    agent = get_news_agent()

    if not st.session_state['news_loaded']:
        with st.spinner("Loading latest news..."):
            news_articles = agent.fetch_news(query = "India stock market Nifty")
            if news_articles:
                agent.create_embeddings(news_articles)
                st.session_state['news_loaded'] = True
                st.session_state['news_articles'] = news_articles

    for msg in st.session_state['messages']:
        role = msg['role']
        content = msg['content']
        if role == 'user':
            st.markdown(f"<div class='chat-message user-message'>{content}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-message assistant-message'>{content}</div>", unsafe_allow_html=True)
        
    user_input = st.text_input("Ask about VaR predictions, market insights, or risk management strategies:")

    if user_input:
        st.session_state['messages'].append({'role': 'user', 'content': user_input})
        news_context = ""
        relevant_news = agent.query_news(user_input, n_results=3)
        if relevant_news:
            news_context = "\n".join([f"- {article['title']} ({article['source']})" for article in relevant_news])
        
        var_context = st.session_state.get('var_context', 'No VaR analysis context available.')
        response = agent.chat_completion(user_input, var_context=var_context, news_context=news_context)

        st.session_state['messages'].append({'role': 'assistant', 'content': response})

        st.rerun()
    if not st.session_state['messages']:
        st.markdown("Quick questions to ask the assistant:")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("What is VaR?"):
                st.session_state['messages'].append({'role': 'user', 'content': "What is VaR and how it is calculated?"})
                st.rerun()
        with col2:
            if st.button("Interpret VaR results"):
                st.session_state['messages'].append({'role': 'user', 'content': "Can you help to interpret VaR results?"})
                st.rerun()
        with col3:
            if st.button("Market Outlook"):
                st.session_state['messages'].append({'role': 'user', 'content': "What is current market outlook based on recent news?"})
                st.rerun()
if __name__ == "__main__":
    main()