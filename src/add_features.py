"""
Use this module to add features from `feature_engineering.py`

All features from this module are imported, and more can be added
(such as taking the difference between two)
"""

import numpy as np
import pandas as pd

from utility_functions import add_dow_dummy

from feature_engineering import *


def add_features(df: pd.DataFrame, session):
    """
    Use this function to add above features to the df. make sure to do all
    necessary preprocessing such as one-hot encoding etc.

    params:
        df (pd.DataFrame) - the dataframe we are adding these features to

        session - SessionInfo object containing information about current
                session info, particularly the number of classes

    returns:
        a new dataframe with all new features already one-hot encoded etc.,
        a list of all feature names to be lagged
    """
    # first, add the target variable according to num_classes
    add_return(df)
    add_log_return(df)

    if session.num_classes == 3:
        add_jump_categories_3(df, session.up_margin, session.down_margin)
    elif session.num_classes == 5:
        add_jump_categories_5(
            df, session.big_down_margin, session.small_down_margin,
            session.small_up_margin, session.big_up_margin
        )
    else:
        raise ValueError(f'Could not add target variable, received num_classes {session.num_classes}')
    
    df['next_jump'] = df['jump'].shift(-1)
    df = pd.get_dummies(df, columns=['jump'], prefix='jump', drop_first=True)

    add_atr(df, period=12)
    add_atr(df, period=24)
    add_atr(df, period=24*5) # 120

    add_ema(df, period=5)
    add_ema(df, period=24)
    add_ema(df, period=24*5) # 120

    add_sma(df, window=5)
    add_sma(df, window=24)
    add_sma(df, window=24*5) # 120

    add_vidya(df, window=5)
    add_vidya(df, window=24)
    add_vidya(df, window=24*5) #120

    add_cmo(df, window=5)
    add_cmo(df, window=12)
    add_cmo(df, window=24)
    add_cmo(df, window=120)

    add_cmf(df, window=5)
    add_cmf(df, window=12)
    add_cmf(df, window=24)
    add_cmf(df, window=120)

    df['atr_24_atr_12'] = df['atr_24'] - df['atr_12']
    df['ema_sma_5'] = df['ema_5'] - df['sma_5']
    df['ema_sma_24'] = df['ema_24'] - df['sma_24']
    df['ema_sma_120'] = df['ema_120'] - df['sma_120']
    df['vidya_ema_5'] = df['vidya_5'] - df['ema_5']
    df['vidya_ema_24'] = df['vidya_24'] - df['ema_24']
    df['vidya_ema_120'] = df['vidya_120'] - df['ema_120']

    add_vwap(df)
    add_sma_feature(df, 'vwap', window=7)
    add_sma_feature(df, 'vwap', window=24)

    df['vwap_price'] = (df['high'] + df['low'] + df['close']) / 3 - df['vwap']
    df['return_log_return'] = df['return'] - df['log_return']
    df['high_over_low'] = df['high'] / df['low']


    add_sma_feature(df, 'volume', window=7)
    add_sma_feature(df, 'volume', window=24)

    add_dow(df)
    add_dow_dummy(df)
    #df = pd.get_dummies(df, columns=['day_of_week'], prefix='dow', drop_first=True)

    cols = [
        'open', 'high', 'low', 'close', 'volume', 'return', 'log_return', 'jump_neutral', 'jump_up',
        'base_asset_volume', 'no_trades', 'taker_buy_vol', 'taker_buy_base_asset_vol',
        'atr_12', 'atr_24', 'atr_120', 'ema_5', 'ema_24', 'ema_120', 'sma_5', 'sma_24', 'sma_120', 'vwap',
        'vwap_price', 'return_log_return', 'high_over_low', 'sma_volume_7', 'sma_volume_24'
    ]

    df = df.dropna()
    for lag in range(1, session.lag_factor+1):
        for col in cols:
            newcol = np.zeros(df.shape[0]) * np.nan
            newcol[lag:] = df[col].values[:-lag]
            df.insert(len(df.columns), "{0}_{1}".format(col, lag), newcol)

    df = df.dropna()

    return df



def add_features_exp(df: pd.DataFrame, session):
    add_return(df)
    add_log_return(df)

    if session.num_classes == 3:
        add_jump_categories_3(df, session.up_margin, session.down_margin)
    elif session.num_classes == 5:
        add_jump_categories_5(
            df, session.big_down_margin, session.small_down_margin,
            session.small_up_margin, session.big_up_margin
        )
    else:
        raise ValueError(f'Could not add target variable, received num_classes {session.num_classes}')
    
    df['next_jump'] = df['jump'].shift(-1)
    df = pd.get_dummies(df, columns=['jump'], prefix='jump', drop_first=True)

    # atr indicator
    add_atr(df, period=3)
    add_atr(df, period=5)
    add_atr(df, period=8)
    add_atr(df, period=10)
    add_atr(df, period=12)
    add_atr(df, period=15)
    add_atr(df, period=20)
    add_atr(df, period=25)
    add_atr(df, period=30)
    add_atr(df, period=35)
    add_atr(df, period=40)

    df['atr_5_3'] = df['atr_5'] - df['atr_3']
    df['atr_8_3'] = df['atr_8'] - df['atr_3']
    df['atr_10_3'] = df['atr_10'] - df['atr_3']
    df['atr_12_3'] = df['atr_12'] - df['atr_3']
    df['atr_15_3'] = df['atr_15'] - df['atr_3']
    df['atr_20_3'] = df['atr_20'] - df['atr_3']
    df['atr_25_3'] = df['atr_25'] - df['atr_3']
    df['atr_30_3'] = df['atr_30'] - df['atr_3']

    df['atr_8_5'] = df['atr_8'] - df['atr_5']
    df['atr_10_5'] = df['atr_10'] - df['atr_5']
    df['atr_12_5'] = df['atr_12'] - df['atr_5']
    df['atr_15_5'] = df['atr_15'] - df['atr_5']
    df['atr_20_5'] = df['atr_20'] - df['atr_5']
    df['atr_25_5'] = df['atr_25'] - df['atr_5']
    df['atr_30_5'] = df['atr_30'] - df['atr_5']

    df['atr_10_8'] = df['atr_10'] - df['atr_8']
    df['atr_12_8'] = df['atr_12'] - df['atr_8']
    df['atr_15_8'] = df['atr_15'] - df['atr_8']
    df['atr_20_8'] = df['atr_20'] - df['atr_8']
    df['atr_25_8'] = df['atr_25'] - df['atr_8']
    df['atr_30_8'] = df['atr_30'] - df['atr_8']

    add_avg_price(df)

    # moving average indicators
    add_ema(df, period=3)
    add_ema(df, period=5)
    add_ema(df, period=8)
    add_ema(df, period=10)
    add_ema(df, period=12)
    add_ema(df, period=15)
    add_ema(df, period=17)
    add_ema(df, period=20)
    add_ema(df, period=25)
    add_ema(df, period=30)

    add_sma(df, window=3)
    add_sma(df, window=5)
    add_sma(df, window=8)
    add_sma(df, window=10)
    add_sma(df, window=12)
    add_sma(df, window=15)
    add_sma(df, window=17)
    add_sma(df, window=20)
    add_sma(df, window=25)
    add_sma(df, window=30)

    add_vidya(df, window=3)
    add_vidya(df, window=5)
    add_vidya(df, window=8)
    add_vidya(df, window=10)
    add_vidya(df, window=12)
    add_vidya(df, window=15)
    add_vidya(df, window=17)
    add_vidya(df, window=20)
    add_vidya(df, window=25)
    add_vidya(df, window=30)

    drop = [
        'ema_3', 'ema_5', 'ema_8', 'ema_8', 'ema_10', 'ema_12', 'ema_15', 'ema_17', 'ema_20', 'ema_30', 
        'sma_3', 'sma_5', 'sma_8', 'sma_8', 'sma_10', 'sma_12', 'sma_15', 'sma_17', 'sma_20', 'sma_30',
        'vidya_3', 'vidya_5', 'vidya_8', 'vidya_8', 'vidya_10', 'vidya_12', 'vidya_15', 'vidya_17', 'vidya_20', 'vidya_30'
    ]

    # difference between moving average and price
    df['price_sma_3'] = df['avg_price'] - df['sma_3']
    df['price_sma_5'] = df['avg_price'] - df['sma_5']
    df['price_sma_8'] = df['avg_price'] - df['sma_8']
    df['price_sma_10'] = df['avg_price'] - df['sma_10']
    df['price_sma_12'] = df['avg_price'] - df['sma_12']
    df['price_sma_15'] = df['avg_price'] - df['sma_15']
    df['price_sma_17'] = df['avg_price'] - df['sma_17']
    df['price_sma_20'] = df['avg_price'] - df['sma_20']
    df['price_sma_30'] = df['avg_price'] - df['sma_30']

    df['price_vidya_3'] = df['avg_price'] - df['vidya_3']
    df['price_vidya_5'] = df['avg_price'] - df['vidya_5']
    df['price_vidya_8'] = df['avg_price'] - df['vidya_8']
    df['price_vidya_10'] = df['avg_price'] - df['vidya_10']
    df['price_vidya_12'] = df['avg_price'] - df['vidya_12']
    df['price_vidya_15'] = df['avg_price'] - df['vidya_15']
    df['price_vidya_17'] = df['avg_price'] - df['vidya_17']
    df['price_vidya_20'] = df['avg_price'] - df['vidya_20']
    df['price_vidya_30'] = df['avg_price'] - df['vidya_30']

    df['price_vidya_3'] = df['avg_price'] - df['vidya_3']
    df['price_vidya_5'] = df['avg_price'] - df['vidya_5']
    df['price_vidya_8'] = df['avg_price'] - df['vidya_8']
    df['price_vidya_10'] = df['avg_price'] - df['vidya_10']
    df['price_vidya_12'] = df['avg_price'] - df['vidya_12']
    df['price_vidya_15'] = df['avg_price'] - df['vidya_15']
    df['price_vidya_17'] = df['avg_price'] - df['vidya_17']
    df['price_vidya_20'] = df['avg_price'] - df['vidya_20']
    df['price_vidya_30'] = df['avg_price'] - df['vidya_30']

    # differences within indicator
    df['ema_5_3'] = df['ema_5'] - df['ema_3']
    df['ema_8_3'] = df['ema_8'] - df['ema_3']
    df['ema_10_3'] = df['ema_10'] - df['ema_3']
    df['ema_12_3'] = df['ema_12'] - df['ema_3']
    df['ema_15_3'] = df['ema_15'] - df['ema_3']
    df['ema_20_3'] = df['ema_20'] - df['ema_3']
    df['ema_25_3'] = df['ema_25'] - df['ema_3']
    df['ema_30_3'] = df['ema_30'] - df['ema_3']

    df['ema_8_5'] = df['ema_8'] - df['ema_5']
    df['ema_10_5'] = df['ema_10'] - df['ema_5']
    df['ema_12_5'] = df['ema_12'] - df['ema_5']
    df['ema_15_5'] = df['ema_15'] - df['ema_5']
    df['ema_20_5'] = df['ema_20'] - df['ema_5']
    df['ema_25_5'] = df['ema_25'] - df['ema_5']
    df['ema_30_5'] = df['ema_30'] - df['ema_5']

    df['ema_10_8'] = df['ema_10'] - df['ema_8']
    df['ema_12_8'] = df['ema_12'] - df['ema_8']
    df['ema_15_8'] = df['ema_15'] - df['ema_8']
    df['ema_20_8'] = df['ema_20'] - df['ema_8']
    df['ema_25_8'] = df['ema_25'] - df['ema_8']
    df['ema_30_8'] = df['ema_30'] - df['ema_8']

    df['sma_5_3'] = df['sma_5'] - df['sma_3']
    df['sma_8_3'] = df['sma_8'] - df['sma_3']
    df['sma_10_3'] = df['sma_10'] - df['sma_3']
    df['sma_12_3'] = df['sma_12'] - df['sma_3']
    df['sma_15_3'] = df['sma_15'] - df['sma_3']
    df['sma_20_3'] = df['sma_20'] - df['sma_3']
    df['sma_25_3'] = df['sma_25'] - df['sma_3']
    df['sma_30_3'] = df['sma_30'] - df['sma_3']

    df['sma_8_5'] = df['sma_8'] - df['sma_5']
    df['sma_10_5'] = df['sma_10'] - df['sma_5']
    df['sma_12_5'] = df['sma_12'] - df['sma_5']
    df['sma_15_5'] = df['sma_15'] - df['sma_5']
    df['sma_20_5'] = df['sma_20'] - df['sma_5']
    df['sma_25_5'] = df['sma_25'] - df['sma_5']
    df['sma_30_5'] = df['sma_30'] - df['sma_5']

    df['sma_10_8'] = df['sma_10'] - df['sma_8']
    df['sma_12_8'] = df['sma_12'] - df['sma_8']
    df['sma_15_8'] = df['sma_15'] - df['sma_8']
    df['sma_20_8'] = df['sma_20'] - df['sma_8']
    df['sma_25_8'] = df['sma_25'] - df['sma_8']
    df['sma_30_8'] = df['sma_30'] - df['sma_8']

    df['vidya_5_3'] = df['vidya_5'] - df['vidya_3']
    df['vidya_8_3'] = df['vidya_8'] - df['vidya_3']
    df['vidya_10_3'] = df['vidya_10'] - df['vidya_3']
    df['vidya_12_3'] = df['vidya_12'] - df['vidya_3']
    df['vidya_15_3'] = df['vidya_15'] - df['vidya_3']
    df['vidya_20_3'] = df['vidya_20'] - df['vidya_3']
    df['vidya_25_3'] = df['vidya_25'] - df['vidya_3']
    df['vidya_30_3'] = df['vidya_30'] - df['vidya_3']

    df['vidya_8_5'] = df['vidya_8'] - df['vidya_5']
    df['vidya_10_5'] = df['vidya_10'] - df['vidya_5']
    df['vidya_12_5'] = df['vidya_12'] - df['vidya_5']
    df['vidya_15_5'] = df['vidya_15'] - df['vidya_5']
    df['vidya_20_5'] = df['vidya_20'] - df['vidya_5']
    df['vidya_25_5'] = df['vidya_25'] - df['vidya_5']
    df['vidya_30_5'] = df['vidya_30'] - df['vidya_5']

    df['vidya_10_8'] = df['vidya_10'] - df['vidya_8']
    df['vidya_12_8'] = df['vidya_12'] - df['vidya_8']
    df['vidya_15_8'] = df['vidya_15'] - df['vidya_8']
    df['vidya_20_8'] = df['vidya_20'] - df['vidya_8']
    df['vidya_25_8'] = df['vidya_25'] - df['vidya_8']
    df['vidya_30_8'] = df['vidya_30'] - df['vidya_8']

    # differences between indicators
    df['ema_sma_3'] = df['ema_3'] - df['sma_3']
    df['ema_sma_5'] = df['ema_5'] - df['sma_5']
    df['ema_sma_8'] = df['ema_8'] - df['sma_8']
    df['ema_sma_10'] = df['ema_10'] - df['sma_10']
    df['ema_sma_12'] = df['ema_12'] - df['sma_12']
    df['ema_sma_15'] = df['ema_15'] - df['sma_15']
    df['ema_sma_17'] = df['ema_17'] - df['sma_17']
    df['ema_sma_20'] = df['ema_20'] - df['sma_20']
    df['ema_sma_30'] = df['ema_30'] - df['sma_30']

    df['ema_vidya_3'] = df['ema_3'] - df['vidya_3']
    df['ema_vidya_5'] = df['ema_5'] - df['vidya_5']
    df['ema_vidya_8'] = df['ema_8'] - df['vidya_8']
    df['ema_vidya_10'] = df['ema_10'] - df['vidya_10']
    df['ema_vidya_12'] = df['ema_12'] - df['vidya_12']
    df['ema_vidya_15'] = df['ema_15'] - df['vidya_15']
    df['ema_vidya_17'] = df['ema_17'] - df['vidya_17']
    df['ema_vidya_20'] = df['ema_20'] - df['vidya_20']
    df['ema_vidya_30'] = df['ema_30'] - df['vidya_30']

    df['sma_vidya_3'] = df['sma_3'] - df['vidya_3']
    df['sma_vidya_5'] = df['sma_5'] - df['vidya_5']
    df['sma_vidya_8'] = df['sma_8'] - df['vidya_8']
    df['sma_vidya_10'] = df['sma_10'] - df['vidya_10']
    df['sma_vidya_12'] = df['sma_12'] - df['vidya_12']
    df['sma_vidya_15'] = df['sma_15'] - df['vidya_15']
    df['sma_vidya_17'] = df['sma_17'] - df['vidya_17']
    df['sma_vidya_20'] = df['sma_20'] - df['vidya_20']
    df['sma_vidya_30'] = df['sma_30'] - df['vidya_30']

    df.drop(drop, inplace=True, axis=1)

    # cash flow / momentum indicators
    add_cmo(df, window=3)
    add_cmo(df, window=5)
    add_cmo(df, window=8)
    add_cmo(df, window=10)
    add_cmo(df, window=12)
    add_cmo(df, window=15)
    add_cmo(df, window=17)
    add_cmo(df, window=20)
    add_cmo(df, window=25)
    add_cmo(df, window=30)

    add_cmf(df, window=3)
    add_cmf(df, window=5)
    add_cmf(df, window=8)
    add_cmf(df, window=10)
    add_cmf(df, window=12)
    add_cmf(df, window=15)
    add_cmf(df, window=17)
    add_cmf(df, window=20)
    add_cmf(df, window=25)
    add_cmf(df, window=30)

    df['cmo_5_3'] = df['cmo_5'] - df['cmo_3']
    df['cmo_8_3'] = df['cmo_8'] - df['cmo_3']
    df['cmo_10_3'] = df['cmo_10'] - df['cmo_3']
    df['cmo_12_3'] = df['cmo_12'] - df['cmo_3']
    df['cmo_15_3'] = df['cmo_15'] - df['cmo_3']
    df['cmo_20_3'] = df['cmo_20'] - df['cmo_3']
    df['cmo_25_3'] = df['cmo_25'] - df['cmo_3']
    df['cmo_30_3'] = df['cmo_30'] - df['cmo_3']

    df['cmo_8_5'] = df['cmo_8'] - df['cmo_5']
    df['cmo_10_5'] = df['cmo_10'] - df['cmo_5']
    df['cmo_12_5'] = df['cmo_12'] - df['cmo_5']
    df['cmo_15_5'] = df['cmo_15'] - df['cmo_5']
    df['cmo_20_5'] = df['cmo_20'] - df['cmo_5']
    df['cmo_25_5'] = df['cmo_25'] - df['cmo_5']
    df['cmo_30_5'] = df['cmo_30'] - df['cmo_5']

    df['cmo_10_8'] = df['cmo_10'] - df['cmo_8']
    df['cmo_12_8'] = df['cmo_12'] - df['cmo_8']
    df['cmo_15_8'] = df['cmo_15'] - df['cmo_8']
    df['cmo_20_8'] = df['cmo_20'] - df['cmo_8']
    df['cmo_25_8'] = df['cmo_25'] - df['cmo_8']
    df['cmo_30_8'] = df['cmo_30'] - df['cmo_8']
 
    df['cmf_5_3'] = df['cmf_5'] - df['cmf_3']
    df['cmf_8_3'] = df['cmf_8'] - df['cmf_3']
    df['cmf_10_3'] = df['cmf_10'] - df['cmf_3']
    df['cmf_12_3'] = df['cmf_12'] - df['cmf_3']
    df['cmf_15_3'] = df['cmf_15'] - df['cmf_3']
    df['cmf_20_3'] = df['cmf_20'] - df['cmf_3']
    df['cmf_25_3'] = df['cmf_25'] - df['cmf_3']
    df['cmf_30_3'] = df['cmf_30'] - df['cmf_3']

    df['cmf_8_5'] = df['cmf_8'] - df['cmf_5']
    df['cmf_10_5'] = df['cmf_10'] - df['cmf_5']
    df['cmf_12_5'] = df['cmf_12'] - df['cmf_5']
    df['cmf_15_5'] = df['cmf_15'] - df['cmf_5']
    df['cmf_20_5'] = df['cmf_20'] - df['cmf_5']
    df['cmf_25_5'] = df['cmf_25'] - df['cmf_5']
    df['cmf_30_5'] = df['cmf_30'] - df['cmf_5']

    df['cmf_10_8'] = df['cmf_10'] - df['cmf_8']
    df['cmf_12_8'] = df['cmf_12'] - df['cmf_8']
    df['cmf_15_8'] = df['cmf_15'] - df['cmf_8']
    df['cmf_20_8'] = df['cmf_20'] - df['cmf_8']
    df['cmf_25_8'] = df['cmf_25'] - df['cmf_8']
    df['cmf_30_8'] = df['cmf_30'] - df['cmf_8']

    # volume indicators
    df['prop_taker'] = df['taker_buy_vol'] = df['volume']
    df['avg_transaction'] = df['volume'] / df['no_trades']

    add_sma_feature(df, 'volume', window=3)
    add_sma_feature(df, 'volume', window=5)
    add_sma_feature(df, 'volume', window=8)
    add_sma_feature(df, 'volume', window=10)
    add_sma_feature(df, 'volume', window=12)
    add_sma_feature(df, 'volume', window=15)
    add_sma_feature(df, 'volume', window=17)
    add_sma_feature(df, 'volume', window=20)
    add_sma_feature(df, 'volume', window=25)
    add_sma_feature(df, 'volume', window=30)

    drop = [
        'sma_volume_3', 'sma_volume_5', 'sma_volume_8', 'sma_volume_8', 'sma_volume_10',
        'sma_volume_12', 'sma_volume_15', 'sma_volume_17', 'sma_volume_20', 'sma_volume_30'
    ]

    df['sma_volume_5_3'] = df['sma_volume_5'] - df['sma_volume_3']
    df['sma_volume_8_3'] = df['sma_volume_8'] - df['sma_volume_3']
    df['sma_volume_10_3'] = df['sma_volume_10'] - df['sma_volume_3']
    df['sma_volume_12_3'] = df['sma_volume_12'] - df['sma_volume_3']
    df['sma_volume_15_3'] = df['sma_volume_15'] - df['sma_volume_3']
    df['sma_volume_20_3'] = df['sma_volume_20'] - df['sma_volume_3']
    df['sma_volume_25_3'] = df['sma_volume_25'] - df['sma_volume_3']
    df['sma_volume_30_3'] = df['sma_volume_30'] - df['sma_volume_3']

    df['sma_volume_8_5'] = df['sma_volume_8'] - df['sma_volume_5']
    df['sma_volume_10_5'] = df['sma_volume_10'] - df['sma_volume_5']
    df['sma_volume_12_5'] = df['sma_volume_12'] - df['sma_volume_5']
    df['sma_volume_15_5'] = df['sma_volume_15'] - df['sma_volume_5']
    df['sma_volume_20_5'] = df['sma_volume_20'] - df['sma_volume_5']
    df['sma_volume_25_5'] = df['sma_volume_25'] - df['sma_volume_5']
    df['sma_volume_30_5'] = df['sma_volume_30'] - df['sma_volume_5']

    df['sma_volume_10_8'] = df['sma_volume_10'] - df['sma_volume_8']
    df['sma_volume_12_8'] = df['sma_volume_12'] - df['sma_volume_8']
    df['sma_volume_15_8'] = df['sma_volume_15'] - df['sma_volume_8']
    df['sma_volume_20_8'] = df['sma_volume_20'] - df['sma_volume_8']
    df['sma_volume_25_8'] = df['sma_volume_25'] - df['sma_volume_8']
    df['sma_volume_30_8'] = df['sma_volume_30'] - df['sma_volume_8']

    df.drop(drop, inplace=True, axis=1)

    # price indicators
    add_vwap(df)

    df['vwap_price'] = df['avg_price'] - df['vwap']

    df['return_log_return'] = df['return'] - df['log_return']
    df['return_over_log_return'] = df['return'] / df['log_return']

    df['high_low'] = df['high'] - df['low']
    df['high_over_low'] = df['high'] / df['low']

    add_dow(df)
    add_dow_dummy(df)

    drop = [
        'open', 'high', 'low', 'close', 'volume', 'base_asset_volume', 'no_trades', 'taker_buy_vol', 'taker_buy_base_asset_vol', 'avg_price'
    ]
    df.drop(drop, inplace=True, axis=1)

    # cols to lag
    cols = [
        'return', 'log_return', 'jump_neutral', 'jump_up', 'return_log_return'
    ]

    df = df.dropna()
    for lag in range(1, session.lag_factor+1):
        for col in cols:
            newcol = np.zeros(df.shape[0]) * np.nan
            newcol[lag:] = df[col].values[:-lag]
            df.insert(len(df.columns), "{0}_{1}".format(col, lag), newcol)

    df = df.dropna()

    return df