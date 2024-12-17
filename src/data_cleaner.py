import pandas as pd
import sys

def add_headers_kline(filename: str):
    """
    adds the headers to each column in a kline dataset from binance, changing the 
    original csv file

    params:
        str: filename - name of the csv file 

    returns:
        nothing 
    """
    col_names = [
        'open_time', 'open', 'high', 'low', 'close', 'close_time',
        'base_asset_volume', 'no_trades', 'taker_buy_vol', 'taker_buy_base_asset_vol', 'ignore'
    ]

    df = pd.read_csv(filename, header=None, names=col_names)
    df = df.drop(columns=df.columns[-1])

    df.to_csv(filename, index=False)

if __name__ == '__main__':
    filename = sys.argv[1]
    add_headers_kline(filename=filename)
