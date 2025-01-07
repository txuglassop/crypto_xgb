import pandas as pd
import sys

def add_headers_kline(path: str):
    """
    adds the headers to each column in a kline dataset from binance, changing the 
    original csv file. also converts open_time from unix time to d/m/y

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
    #df['date'] = pd.to_datetime(df['open_time'], unit='ms').dt.strftime('%d/%m/%Y')
    df['date'] = df['open_time']
    df = df.drop(columns=['ignore', 'open_time', 'close_time'])

    cols = ['date'] + [col for col in df.columns if col != 'date']
    df = df[cols]

    df.to_csv(path, index=False)

if __name__ == '__main__':
    path = sys.argv[1]
    add_headers_kline(path=path)
