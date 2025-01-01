import pandas as pd
import numpy as np

def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe with open, high, low, close, volume features, calculates the
    VWAP indicator for each row. Returns a new dataframe with a new column containing
    this

    params:
        pd DataFrame: df - a dataframe containing features as described above

    returns:
        pd DataFrame - a new df containing this new feature 
    """
    df['VWAP'] = (((df['high'] + df['low'] + df['close']) / 3) * df['volume']).cumsum() / df['volume'].cumsum()

    return df

def add_ema(df: pd.DataFrame, period = 5, weighting_factor = 0.2) -> pd.DataFrame:
    """
    Given a dataframe, adds Exponential Moving Average (EMA) with specified period and weighting
    factor. 

    params:
        pd DataFrame: df - a dataframe containing 'high', 'low', 'close'
        int: period - specifying the period the EMA is taken over
        float: weighting_factor - a number in (0,1) that specfies the weight placed on recent obs

    returns:
        pd DataFrame - a new df containing this new feature. Note NA's will be introduced
    """
    prices = np.divide(np.add(np.add(df['high'], df['low']), df['close']),3)
    ema = np.zeros(len(prices))
    sma = np.mean(prices[:period])
    ema[period - 1] = sma
    for idx in range(period, len(prices)):
        ema[idx] = (prices[idx] * weighting_factor) + (ema[idx - 1] * (1 - weighting_factor))

    df['ema'] = ema
    df['ema'] = df['ema'].replace(0, pd.NA)

    return df

def add_atr(df: pd.DataFrame, period = 14) -> pd.DataFrame:
    """
    Given a dataframe, adds Average True Range (ATR) according to a specified period.

    params:
        pd DataFrame: df - a dataframe containing 'high', 'low', 'close'
        int: period - specifies the period the ATR is taken over

    returns:
        pd DataFrame - a new df containing this feature. Note NA's will be introduced
    """
    high = df['high']
    low = df['low']
    close = df['close']

    tr = np.zeros(len(high))
    for idx in range(1, len(high)):
        H = high[idx]
        L = low[idx]
        C_p = close[idx - 1]
        tr[idx] = max(max(abs(H-L), abs(H-C_p)), abs(L-C_p))

    atr = np.zeros(len(tr))
    atr[period] = np.mean(tr[1:period-1])
    for idx in range(period+1, len(tr)):
        atr[idx] = (atr[idx-1] + tr[idx]) / period

    df['atr'] = atr
    df['atr'] = df['atr'].replace(0, pd.NA)

    return df

def add_dow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe with a 'date' column in either '%d/%m/%y' or UNIX time (int), adds
    a new column with the day of the week

    params:
        pd.DataFrame: df - dataframe as described above

    returns:
        pd.DataFrame - the same dataframe with a new column 'day_of_week' of type string
    """
    if df.dtypes['date'] == int or df.dtypes['date'] == float:
        df['date'] = pd.to_datetime(df['date'], unit='ms').dt.strftime('%d/%m/%Y')
    
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

    df['day_of_week'] = df['date'].dt.day_name()

    return df