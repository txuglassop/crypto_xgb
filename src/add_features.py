"""
Use this module to add features from `feature_engineering.py`

All features from this module are imported, and more can be added
(such as taking the difference between two)
"""

import numpy as np
import pandas as pd

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

    #add_cmf(df, window=5)
    #add_cmf(df, window=12)
    #add_cmf(df, window=24)
    #add_cmf(df, window=120)

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


    add_sma_feature(df, 'volume', window=7)
    add_sma_feature(df, 'volume', window=24)

    add_dow(df)
    df = pd.get_dummies(df, columns=['day_of_week'], prefix='dow', drop_first=True)

    cols = [
        'open', 'high', 'low', 'close', 'volume', 'return', 'log_return', 'jump_neutral', 'jump_up',
        'atr_12', 'atr_24', 'atr_120', 'ema_5', 'ema_24', 'ema_120', 'sma_5', 'sma_24', 'sma_120', 'vwap',
        'vwap_price', 'return_log_return', 'sma_volume_7', 'sma_volume_24'
    ]

    return df, cols