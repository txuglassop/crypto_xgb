import pandas as pd
import time
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from datetime import datetime
from zoneinfo import ZoneInfo
from binance.client import Client

from user_info import API, USER
from check_data import check_data
from get_data import get_data
from check_data import check_data

def update_data(path: str):
    df = pd.read_csv(path)
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    print('\nRunning data check BEFORE adding any observations\n')
    check_data(df)

    data = path.split('/')[-1]
    symbol = data.split('_')[0]
    interval = data.split('_')[1]

    dt = datetime.fromtimestamp(df['timestamp'].iloc[-1] / 1000)
    year, month, day = dt.year, dt.month, dt.day

    print('\nData check on NEW data\n')
    new_obs = get_data(symbol, interval, year, month, day)
    new_df = pd.concat([df, new_obs])

    new_df.drop_duplicates(['timestamp'], inplace=True)
    new_df.sort_values(['timestamp'], inplace=True)

    print('\nData check on UPDATED data\n')
    check_data(new_df)

    while True:
        c = input('Would you like to overwrite the original data with this new df? [Y/n]: ').lower()
        if c == 'n':
            exit(1)
        elif c == 'y':
            break
    
    new_df.to_csv(path)


if __name__ == '__main__':
    path = sys.argv[1]
    update_data(path)
