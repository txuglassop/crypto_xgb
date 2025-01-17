import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_backtest(backtest_results: pd.DataFrame, portion = [0, 100]):
    """
    Visualises backtest results, plotting a certain portion of the test

    params:
        backtest_results (pd.DataFrame) - the results returned from `run_backtest`
                from the `Backtest` class

        portion (list) - an array of len 2 that describes the portion of the 
                backtest to be plotted
    """
    if len(portion) != 2 or portion[1] <= portion[0]:
        raise ValueError('portion argument inappropriate.')
    
    start_index = int(portion[0] / 100 * backtest_results.shape[0])
    end_index = int(portion[1] / 100 * backtest_results.shape[0])
    df = backtest_results.iloc[start_index:end_index]
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.4, 0.3, 0.3], 
                        subplot_titles=("OHLC Chart of Test Data", "Portfolio Value", "Capital Over Time"))

    # OHLC plot
    fig.add_trace(go.Ohlc(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    ), row=1, col=1)

    portfolio_value = df['capital'] + df['trades'].cumsum() * df['close']    

    # portfolio value plot
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=portfolio_value,
        mode='lines',
        line=dict(color='blue'),
        name='Current Portfolio Value'
    ), row=2, col=1)

    # capital over time plot
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['capital'],
        mode='lines',
        line=dict(color='purple'),
        name='Current Capital'
    ), row=3, col=1)

    fig.update_layout(
        title="Backtest Results with Trades and Capital",
        xaxis=dict(rangeslider=dict(visible=False)),
        template="plotly_dark",
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value", row=2, col=1)
    fig.update_yaxes(title_text="Capital", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)

    
    fig.show()