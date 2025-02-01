import pandas as pd
import numpy as np
import sys

def add_headers_kline(path: str):
    """
    adds the headers to each column in a kline dataset from binance, changing the 
    original csv file. note `open_time` is renamed to `timestamp`

    params:
        str: path - path to the csv

    returns:
        nothing 
    """
    col_names = [
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'base_asset_volume', 'no_trades', 'taker_buy_vol', 'taker_buy_base_asset_vol', 'ignore'
    ]

    df = pd.read_csv(path, header=None, names=col_names, index_col=False)
    df['timestamp'] = df['open_time']
    df = df.drop(columns=['ignore', 'open_time', 'close_time'])

    cols = ['timestamp'] + [col for col in df.columns if col != 'timestamp']
    df = df[cols]

    df.to_csv(path, index=False)

def sort_and_check_data(path: str):
    """
    ensures data is ordered in ascending order of the `timestamp` feature and removes any duplicates.
    will then print all unique increments in `timestamp` as a way of confirming there are no missing
    rows. any irregular / outstanding increments should be investigated

    params:
        path (str) - path to the csv
    """
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)
    df.sort_values('timestamp', inplace=True)

    incr = np.diff(df['timestamp'])
    unique, counts = np.unique(incr, return_counts=True)

    for u,c in zip(unique, counts):
        print(f'Incr: {u}, Count: {c}')

if __name__ == '__main__':
    path = sys.argv[1]
    add_headers_kline(path=path)
    sort_and_check_data(path=path)