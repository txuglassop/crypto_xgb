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


lookup = {
    '1m': Client.KLINE_INTERVAL_1MINUTE,
    '3m': Client.KLINE_INTERVAL_3MINUTE,
    '15m': Client.KLINE_INTERVAL_15MINUTE,
    '30m': Client.KLINE_INTERVAL_30MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR
}

def get_data(symbol, interval, year, month, day):
    client = Client(API['API_KEY'], API['API_SECRET'])
    tz = ZoneInfo(USER['TIMEZONE'])

    end_time = datetime.fromtimestamp(time.time())
    # format (year, month, day, hour, minute, second)
    start_time = datetime(year, month, day, 0, 0, 0, tzinfo=tz)

    client_interval = lookup[interval]

    klines = client.get_historical_klines(symbol=symbol, interval=client_interval, start_str=str(start_time), end_str=str(end_time))
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'Close Time', 'base_asset_volume', 'no_trades', 'taker_buy_vol', 'taker_buy_base_asset_vol', 'Ignore'])
    df.drop(['Close Time', 'Ignore'], axis=1, inplace=True)

    # drop last row since candle is not closed yet
    df = df.iloc[:-1]
    check_data(df)

    name = symbol + '_' + interval + '_' + str(year)
    if month != 1:
        name = name + '_' + str(month)
    if day != 1:
        name = name + '_' + str(day)

    df.to_csv('./input/' + name + '.csv')

if __name__ == '__main__':
    symbol = sys.argv[1].upper()
    interval = sys.argv[2]
    year = int(sys.argv[3])
    month = int(sys.argv[4]) 
    day = int(sys.argv[5])

    get_data(symbol, interval, year, month, day)

