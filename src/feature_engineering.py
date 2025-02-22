"""
To add features, add a function that takes in `df` (the dataframe with OHLC vol data)
which creates a new column with this feature.

Then, using the bottom function `add_features`, add a function call with `df` to add the 
feature, making sure that any prerequisite features are there already. also add the name
of the feature to the `cols` if the feature is to be lagged.

NOTE: Do NOT drop any na - anything calling `add_features` will drop na's when needed
"""

import pandas as pd
import numpy as np

def add_return(df: pd.DataFrame):
    """
    Given a dataframe with OHLC data, calculates percentage change in average price i.e. the return
    
    params:
        pd.DataFrame: df - dataframe as described above
    """
    prices = np.divide(np.add(np.add(df['high'], df['low']), df['close']),3)
    returns = np.zeros(len(prices))
    returns[0] = np.nan

    for idx in range(1, len(prices)):
        returns[idx] = prices.iloc[idx] / prices.iloc[idx - 1] - 1
    
    df['return'] = returns

def add_log_return(df: pd.DataFrame):
    prices = np.divide(np.add(np.add(df['high'], df['low']), df['close']),3)
    prices = prices.to_numpy()
    log_returns = np.empty_like(prices, dtype=float)
    log_returns[0] = np.nan
    log_returns[1:] = np.log(prices[1:] / prices[:-1])

    df['log_return'] = log_returns

def add_avg_price(df: pd.DataFrame):
    prices = np.divide(np.add(np.add(df['high'], df['low']), df['close']),3)
    df['avg_price'] = prices

def add_jump_categories_3(df: pd.DataFrame, up_margin = 0.025, down_margin = 0.025):
    """
    Given a dataframe with a 'return' feature, adds a new feature which categorises the returns
    into 3 categories according to the provided margin (around a return of 0, which is neutral)

    params:
        pd.DataFrame: df - dataframe with return col, ideally created by using add_return
        float: margin - margin required when a return is considered significant enough to warrant
            a category

        float: up_margin - if a return is greater than this, classifed as 'up'

        float: down_margin - if a return is less than this, classified as 'down'.
    """
    def jump_lookup(x):
        if x < -np.abs(down_margin):
            return 'down'
        elif x > up_margin:
            return 'up'
        else:
            return 'neutral'
        
    jump = df['return']
    jump = jump.apply(jump_lookup)

    df['jump'] = jump


def add_jump_categories_5(df: pd.DataFrame, big_down_margin = 0.025, small_down_margin = 0.01,
                          small_up_margin = 0.01, big_up_margin = 0.025):
    """
    Given a dataframe with a 'return' feature, adds a new feature which categorises the returns
    into 5 categories according to the provided margins, with 'small' jumps being returns
    are between the small and big margin, and big jumps are returns greater than the big_margin

    params:
        pd.DataFrame: df - dataframe with the return col

        all margins are floats, taken as absolute values, and interpreted the same as
        `add_jump_categories_3`
    """

    def jump_lookup(x):
        if x > big_up_margin:
            return 'big_up'
        elif x < - np.abs(big_down_margin):
            return 'big_down'
        elif x > small_up_margin:
            return 'small_up'
        elif x < -np.abs(small_down_margin):
            return 'small_down'
        else:
            return 'neutral'
        
    jump = df['return']
    jump = jump.apply(jump_lookup)

    df['jump'] = jump


def add_vwap(df: pd.DataFrame):
    """
    Given a dataframe with open, high, low, close, volume features, calculates the
    VWAP indicator for each row. Returns a new dataframe with a new column containing
    this

    params:
        pd DataFrame: df - a dataframe containing features as described above
    """
    df['vwap'] = (((df['high'] + df['low'] + df['close']) / 3) * df['volume']).cumsum() / df['volume'].cumsum()

def add_ema(df: pd.DataFrame, period = 5, weighting_factor = 0.2):
    """
    Given a dataframe, adds Exponential Moving Average (EMA) with specified period and weighting
    factor. 

    params:
        pd DataFrame: df - a dataframe containing 'high', 'low', 'close'
        int: period - specifying the period the EMA is taken over
        float: weighting_factor - a number in (0,1) that specfies the weight placed on recent obs
    """
    prices = np.divide(np.add(np.add(df['high'], df['low']), df['close']),3)
    ema = np.zeros(len(prices))
    sma = np.mean(prices[:period])
    ema[period - 1] = sma
    for idx in range(period, len(prices)):
        ema[idx] = (prices.iloc[idx] * weighting_factor) + (ema[idx - 1] * (1 - weighting_factor))

    name = 'ema_' + str(period)

    df[name] = ema
    df[name] = df[name].replace(0, np.nan)

def add_sma(df: pd.DataFrame, window = 7):
    """
    Adds a simple moving average feature, taking `window` entries into account

    params:
        df (pd.DataFrame)
        window (int) - the number of entries to take the average of. includes the
            current row and the past `window-1` entries.
    """
    prices = np.divide(np.add(np.add(df['high'], df['low']), df['close']),3)
    sma = prices.rolling(window=window).mean()

    name = 'sma_' + str(window)

    df[name] = sma
    df[name] = df[name].replace(0, np.nan)
        
def add_sma_feature(df: pd.DataFrame, feature_name: str, window = 7):
    """
    Adds a simple moving average for a feature already in `df` 

    params:
        df (pd.DataFrame)
        feature_name (str): the name of the feature as a string
        window (int) - the number of entries to take the average of. includes the
            current row and the past `window-1` entries.
    """
    feature = df[feature_name]
    sma = feature.rolling(window=window).mean()

    name = 'sma_' + feature_name + '_' + str(window)

    df[name] = sma
    df[name] = df[name].replace(0, np.nan)

def add_atr(df: pd.DataFrame, period = 14):
    """
    Given a dataframe, adds Average True Range (ATR) according to a specified period.

    params:
        pd DataFrame: df - a dataframe containing 'high', 'low', 'close'
        int: period - specifies the period the ATR is taken over
    """
    high = df['high']
    low = df['low']
    close = df['close']

    tr = np.zeros(len(high))
    for idx in range(1, len(high)):
        H = high.iloc[idx]
        L = low.iloc[idx]
        C_p = close.iloc[idx - 1]
        tr[idx] = max(max(abs(H-L), abs(H-C_p)), abs(L-C_p))

    atr = np.zeros(len(tr))
    atr[period] = np.mean(tr[1:period-1])
    for idx in range(period+1, len(tr)):
        atr[idx] = (atr[idx-1] + tr[idx]) / period

    name = 'atr_' + str(period)
    df[name] = atr
    df[name] = df[name].replace(0, np.nan)

def add_dow(df: pd.DataFrame):
    """
    Given a dataframe with a 'timestamp' column in either '%d/%m/%y' or UNIX time (int), adds
    a new column with the day of the week

    params:
        pd.DataFrame: df - dataframe as described above
    """
    time = df['timestamp']

    if df.dtypes['timestamp'] == int or df.dtypes['timestamp'] == float:
        time = pd.to_datetime(df['timestamp'], unit='ms').dt.strftime('%d/%m/%Y')
    
    time = pd.to_datetime(time, format='%d/%m/%Y')

    df['day_of_week'] = time.dt.day_name()

def add_vidya(df: pd.DataFrame, window: int, alpha_min=0.05, alpha_max=0.25):
    """
    adds Variance Index Dynamic Average (VIDYA) indicator, using standard deviation
    defined by `window` as variance indicator.
    https://lightningchart.com/blog/trader/variable-index-dynamic-average
    adding differences between this feature and other average price indiciators
    with different windows can be a good volatility indicator

    params:
        df (pd.DataFrame) - dataframe must have `ema_{window}` feature
        window - window to be taken over
    """
    rolling_std = np.zeros(len(df))
    prices = np.divide(np.add(np.add(df['high'], df['low']), df['close']),3)
    for i in range(window-1, len(prices)):
        rolling_std[i] = np.std(prices[i-window+1:i+1])

    s_max = np.max(rolling_std)
    s_min = np.min(rolling_std[rolling_std > 0])
    print(f's_max = {s_max}, s_min = {s_min}')

    alpha = alpha_min * ( (rolling_std - s_min) / (s_max - s_min) ) * (alpha_max - alpha_min)

    ema_name = 'ema_' + str(window)
    vidya = df[ema_name] + alpha * (prices - df[ema_name])
    name = 'vidya_' + str(window)
    df[name] = vidya


def add_cmo(df: pd.DataFrame, window: int):
    """
    Adds Chande Momentum Oscillator (CMO) indicator, using the past `window` closes.
    https://www.investopedia.com/terms/c/chandemomentumoscillator.asp
    Include several different windows for this period, and also differences between

    The new feature has name `f'cmo_{window}`.

    params:
        df (pd.DataFrame) - dataframe must have `close` feature
        window (int) - window to take closes over
    """
    close_diffs = df['close'].diff()
    m1 = np.where(close_diffs >= 0, close_diffs, 0.0)
    m1[0] = np.nan
    m2 = np.where(close_diffs >= 0, 0.0, -close_diffs)
    
    sm1 = pd.Series(m1).rolling(window).sum()
    sm2 = pd.Series(m2).rolling(window).sum()

    cmo = 100 * (sm1-sm2) / (sm1+sm2)

    name = 'cmo_' + str(window) 
    df[name] = np.array(cmo)
    #df[name] = df[name].replace(0, np.nan)

def add_alma(df: pd.DataFrame, window: int = 9, sigma: float = 6, offset: float = 0.85):
    """
    Adds Arnaud Legoux Moving Average (ALMA) to `df.` `df` must have `avg_price` feature

    The name of the new feature is `f'alma_{window}_{sigma}_{offset}`
    
    Ref: https://stackoverflow.com/a/76031340/
    """
    prices = np.array(df['avg_price'])
    m = offset * (window - 1)
    s = window / sigma
    i = np.arange(window)
    weights = np.exp(-1 * np.square(i - m) / (2 * np.square(s)))
    norm_weights = weights / np.sum(weights)
    padded_prices = np.pad(prices, (window - 1, 0), mode='edge')
    alma_values = np.convolve(padded_prices, norm_weights[::-1], mode='valid')

    name = 'alma_' + str(window) + '_' + str(sigma) + '_' + str(offset)
    df[name] = alma_values
    #df[name] = df[name].replace(0, np.nan)

def add_cmf(df, window: int=5):
    """
    Adds the Chaikin Money Flow (CMO) indicator to `df`, using taking the past
    `window` periods into account

    The name of the new feature is `f'cmf_{window}`

    params:
        df (pd.DataFrame) - as usual
        window (int)
    """
    mfv = df['volume'] * (2*df['close'] - df['high'] - df['low']) / \
        (df['high'] - df['low'])
    
    mfv_series = pd.Series(mfv).rolling(window).sum()
    cmf = mfv_series / df['volume'].rolling(window).sum()

    name = 'cmf_' + str(window)
    df[name] = cmf

def add_rsi(df: pd.DataFrame, window: int=5):
    change = df['close'].diff()
    change_up = change.copy()
    change_down = change.copy()

    change_up[change_up<0] = 0
    change_down[change_down>0] = 0

    avg_up = change_up.rolling(window).mean()
    avg_down = change_down.rolling(window).mean()

    rsi = 100 * avg_up / (avg_up + avg_down)

    name = 'rsi_' + str(window)
    df[name] = rsi