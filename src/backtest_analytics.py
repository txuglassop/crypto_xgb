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
    if len(portion) != 2 or portion[1] <= portion[0] or portion[0] < 0 or portion[1] > 100:
        raise ValueError('portion argument inappropriate.')
    
    start_index = int(portion[0] / 100 * backtest_results.shape[0])
    end_index = int(portion[1] / 100 * backtest_results.shape[0])
    df = backtest_results.iloc[start_index:end_index]
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.4, 0.3, 0.3], 
                        subplot_titles=("OHLC Chart of Test Data", "Portfolio Value", "Position"))

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

    # trades plot
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['trades'].cumsum(),
        mode='lines',
        line=dict(color='purple'),
        name='Trades'
    ), row=3, col=1)

    fig.update_layout(
        title="Backtest Results with Trades and Capital",
        xaxis=dict(rangeslider=dict(visible=False)),
        template="plotly_dark",
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value", row=2, col=1)
    fig.update_yaxes(title_text="Position", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    
    fig.show()

def trade_analyser(backtest_results: pd.DataFrame):
    """
    Given the backtest results (as returned from `run_backtest`), will
    analyse all trades made throughout the test and return analytics
    about them
    """
    col_names = ['trade_entry', 'trade_exit', 'entry_price', 'exit_price']
    trades_df = pd.DataFrame(columns=col_names)

    # keep track of which trades we have exited out of
    exits = np.zeros(backtest_results.shape[0])
    ongoing_trades = False

    for idx1, row1 in backtest_results.iterrows():
        # if there are ongoing trades, break 
        if ongoing_trades:
            break

        # if there were no buys, continue
        if row1['trades'] <= 0:
            continue
        
        for _ in range(int(row1['trades'])):
            # go through each of the trades made here and find next sell
            for idx2, row2 in backtest_results[idx1:].iterrows():
                if ongoing_trades:
                    break
                elif row2['trades'] < 0 and exits[idx2] != -row2['trades']:
                    # we are exiting this trade - log it!
                    exits[idx2] += 1
                    trades_df.loc[len(trades_df)] = [
                        row1['timestamp'], row2['timestamp'], row1['open'], row2['close']
                    ]
                    break
                elif idx2 == len(exits):
                    # we have reached the end of our test without exiting
                    # all future buys are still ongoing
                    ongoing_trades = True

    return trades_df
