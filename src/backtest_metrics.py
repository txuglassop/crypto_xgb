import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime
from statistics import variance
from textwrap import dedent

from utility_functions import get_monthly_returns, classification_summary

def plot_backtest(backtest_results: pd.DataFrame, portion = [0, 100]) -> None:
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
                        subplot_titles=("OHLC Chart of Test Data", "Equity", "Position"))

    # OHLC plot
    fig.add_trace(go.Ohlc(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    ), row=1, col=1)

    equity = df['capital'] + df['trades'].cumsum() * df['close']    

    # portfolio value plot
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=equity,
        mode='lines',
        line=dict(color='blue'),
        name='Equity'
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

def get_trades(backtest_results: pd.DataFrame) -> pd.DataFrame:
    """
    Given the backtest results (as returned from `run_backtest`), will
    analyse all trades made throughout the test and return analytics
    about them

    params:
        backtest_results (pd.DataFrame) - the results returned from the 
                `run_backtest` module from `backtest`

    returns:
        pd.DataFrame - a dataframe with entry/exit time+price for all trades
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

def get_backtest_metrics(backtest_results: pd.DataFrame, rf_rate = 0.04) -> dict:
    """
    Get metrics relevant to performance of strategy in backtest.
    Inspired by `backtesting.py` backtest output 
    """
    trades_df = get_trades(backtest_results)
    winning_trades = trades_df[trades_df['exit_price'] > trades_df['entry_price']]
    losing_trades = trades_df[trades_df['exit_price'] <= trades_df['entry_price']]
    equity = backtest_results['equity'].values

    # deterministic information
    start = datetime.fromtimestamp(backtest_results['timestamp'][0] / 1000)
    end = datetime.fromtimestamp(backtest_results.iloc[-1]['timestamp'] / 1000)

    time_diff = end - start
    days = time_diff.days
    seconds = time_diff.seconds
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    duration = f'{days} days, {hours} hours, {minutes} mins, {seconds} seconds'

    # equity information
    exposure = np.sum(backtest_results['trades'].cumsum() != 0) / backtest_results.shape[0]
    equity_start = equity[0]
    equity_final = equity[len(equity) - 1]
    equity_high = max(equity)
    equity_low = min(equity)

    buy_hold_ann_return = ((backtest_results.iloc[-1]['close'] /  backtest_results['open'][0]) - 1) / time_diff.days * 365

    monthly_returns = get_monthly_returns(backtest_results['timestamp'], equity)
    avg_ann_return = np.mean(monthly_returns) * 12
    avg_ann_volatility = np.sqrt(variance(monthly_returns) * 12)
    sharpe_ratio = avg_ann_return / avg_ann_volatility

    # trades information
    num_trades = trades_df.shape[0]
    win_rate = winning_trades.shape[0] / num_trades
    avg_trade = np.average(trades_df['exit_price'] / trades_df['entry_price'] - 1)

    winning_trade_returns = winning_trades['exit_price'] / winning_trades['entry_price'] - 1
    losing_trade_returns = losing_trades['exit_price'] / losing_trades['entry_price'] - 1
    best_trade = max(winning_trade_returns)
    avg_winning_trade = np.mean(winning_trade_returns)
    worst_trade = min(losing_trade_returns)
    avg_losing_trade = np.mean(losing_trade_returns)

    time_between_trades = trades_df['trade_exit'] - trades_df['trade_entry']
    avg_trade_duration = np.mean(time_between_trades)
    avg_trade_duration = avg_trade_duration / 1000 / 60 / 60
    avg_trade_duration = str(avg_trade_duration) + ' hours'

    backtest_metrics = {
        'start': start,
        'end': end,
        'duration': duration,
        'exposure': exposure,
        'equity_start': equity_start,
        'equity_high': equity_high,
        'equity_low': equity_low,
        'equity_final': equity_final,
        'buy_hold_ann_return': buy_hold_ann_return,
        'ann_return': avg_ann_return,
        'ann_volatility': avg_ann_volatility,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_trade': avg_trade,
        'avg_trade_duration': avg_trade_duration,
        'best_trade': best_trade,
        'avg_winning_trade': avg_winning_trade,
        'worst_trade': worst_trade,
        'avg_losing_trade': avg_losing_trade
    }

    return backtest_metrics

def print_backtest_metrics(backtest_results: pd.DataFrame, rf_rate = 0.04) -> None:
    metrics = get_backtest_metrics(backtest_results, rf_rate)

    print(f'''
================ Backtest  Metrics ================

---------------- Test  Information ----------------

Start:                          {metrics['start']}
End:                            {metrics['end']}
Duration:                       {metrics['duration']}

---------------- Strategy Metrics -----------------

Strategy:                       {backtest_results.__name__}

Exposure:                       {metrics['exposure']:<10}
Initial Equity:                 {metrics['equity_start']:<10.2f}
Equity High:                    {metrics['equity_high']:<10.2f}
Equity Low:                     {metrics['equity_low']:<10.2f}
Final Equity:                   {metrics['equity_final']:<10.2f}

Return (ann.) [%]:              {metrics['ann_return'] * 100:<10.2f}
Buy & Hold Return (ann.) [%]:   {metrics['buy_hold_ann_return'] * 100:<10.2f}
Volatility (ann.) [%]:          {metrics['ann_volatility'] * 100:<10.2f}
Sharpe Ratio (R={rf_rate*100:.2f}%): {metrics['sharpe_ratio']:<10.2f}

----------------- Trade  Metrics ------------------

Num. Trades:                    {metrics['num_trades']:<10}
Win Rate [%]:                   {metrics['win_rate'] * 100:<10.2f}
Avg. Trade [%]:                 {metrics['avg_trade'] * 100:<10.2f}
Avg. Trade Duration:            {metrics['avg_trade_duration']:<10}

Best Trade [%]:                 {metrics['best_trade'] * 100:<10.2f}
Avg. Winning Trade [%]:         {metrics['avg_winning_trade'] * 100:<10.2f}

Worst Trade [%]:                {metrics['worst_trade'] * 100:<10.2f}
Avg. Losing Trade [%]:          {metrics['avg_losing_trade'] * 100:<10.2f}
''')

    classification_summary(backtest_results['actual'], backtest_results['predictions'])
    

