"""
this executes the wf described in `notebooks/wf.ipynb` using settings from `settings.py`.
results of the backtest will be stored in `backtests/`
"""

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

import sys
import os
from pathlib import Path
repo_path = Path("..")

from settings import data_info, feature_settings

# src imports
from src.session_info import SessionInfo
from src.add_features import add_features
from src.utility_functions import get_jump_lookup
from src.optimise_xgb import stepwise_optimisation

def main():
    data = data_info['data']
    test_size = data_info['test_size']

    num_classes = feature_settings['num_classes']
    lag_factor = feature_settings['lag_factor']
    if num_classes == 3:
        down_margin, up_margin = feature_settings['margins'][num_classes]
        session = SessionInfo(
            data, num_classes, lag_factor, test_size,
            up_margin=up_margin, down_margin=down_margin
        )
    elif num_classes == 5:
        big_down_margin, small_down_margin, small_up_margin, big_up_margin = feature_settings['margins'][num_classes]
        session = SessionInfo(
            data, num_classes, lag_factor, test_size,
            big_down_margin=big_down_margin, small_down_margin=small_down_margin,
            small_up_margin=small_up_margin, big_up_margin=big_up_margin
        )
    else:
        raise ValueError('Number of classes in settings.py not supported')
    
    # data processing
    df = pd.read_csv("input/" + data)
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    # feature engineering
    cols = add_features(df, session)
    df = pd.get_dummies(df, columns=['day_of_week'], prefix='dow', drop_first=True)
    df.dropna()


    for lag in range(1, lag_factor+1):
        for col in cols:
            newcol = np.zeros(df.shape[0]) * np.nan
            newcol[lag:] = df[col].values[:-lag]
            df.insert(len(df.columns), "{0}_{1}".format(col, lag), newcol)

    df = df.dropna()
    df = pd.get_dummies(df, columns=['jump'], prefix='jump', drop_first=True)
    df = df[[col for col in df.columns if col not in ['next_jump']] + ['next_jump']]
    features = [col for col in df.columns if col not in ['timestamp', 'next_jump', 'open', 'high', 'low', 'close', 'volume']]
    session.add_features(features)

    # data preprocessing
    jump_lookup = get_jump_lookup(num_classes)
    X = df.drop(['timestamp', 'next_jump'], axis=1).copy()
    y = df['next_jump'].copy()
    y = y.map(jump_lookup)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    time = df['timestamp'].copy()
    train_timestamp, test_timestamp = time[:len(y_train)], time[len(y_train):]

    # find optimal parameters from the training set







if __name__ == '__main__': main()