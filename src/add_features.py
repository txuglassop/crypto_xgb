"""
Use this module to add features from `feature_engineering.py`

All features from above module are imported, and more can be added
(such as taking the difference between two)
"""

import numpy as np
import pandas as pd

from feature_engineering import *


def add_features(df: pd.DataFrame, session) -> list:
    """
    Use this function to add above features to the df

    params:
        df (pd.DataFrame) - the dataframe we are adding these features to

        session - SessionInfo object containing information about current
                session info, particularly the number of classes

    returns:
        a list of all feature names to be lagged
    """
    # first, add the target variable according to num_classes
    add_return(df)

    if session.num_classes == 3:
        add_jump_categories_3(df, session.up_margin, session.down_margin)
    elif session.num_classes == 5:
        add_jump_categories_5(
            df, session.big_down_margin, session.small_down_margin,
            session.small_up_margin, session.big_up_margin
        )
    else:
        raise(f'Could not add target variable, received num_classes {session.num_classes}')
    
    df['next_jump'] = df['jump'].shift(-1)

    add_atr(df)
    add_ema(df)
    add_vwap(df)
    add_dow(df) # one-hot encoding will be done elsewhere

    cols = ['open', 'high', 'low', 'close', 'volume', 'atr', 'ema', 'vwap']

    return cols