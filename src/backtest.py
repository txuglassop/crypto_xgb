import pandas as pd
import numpy as np
import xgboost as xgb

from typing import Callable, Any
from utility_functions import get_jump_lookup 

class Backtest():
    def __init__(
        self, 
        model_trainer: Callable[[np.ndarray, np.ndarray], Any], 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_test: np.ndarray,
        y_test: np.ndarray,
        num_classes: int,
        starting_capital: float,
        commission: float
    ):
        """
        params:
            model_trainer - a Callable that takes `X_train`, `y_train`, and returns
                    a model that has the `.predict` method
            
            X_train, X_test, y_train, y_test are as usual, and must be of the form
                    that will work if passed into `model_trainer`

            int: num_classes - the number of classes in the target variable

            float: starting capital - your starting capital! used in backtests

            float: commission - the broker commission - used in backtests
        """
        self.train = model_trainer
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.num_classes = num_classes
        self.starting_capital = starting_capital
        self.commission = commission

    def run_backtest(self, strategy: Callable[[int, float, int, float], int],
                     retrain = True, train_frequency = 7):
        """
        Run a backtest using the provided variables when the class was declared and according to
        a provided strategy. `strategies.py` has some default strategies, otherwise they can be
        declared and passed in.

        params:
            strategy - as outlined in `strategies.py`

            boolean: retrain - set to `True` if the model is to be retrained on test data
                throughout the backtest.

            int: train_frequency - how often we retrain our model on incoming observations
        """
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        # make sure the target is of the following form
        jump_lookup = get_jump_lookup(self.num_classes)
        #y_train = y_train.map(jump_lookup)
        #y_test = y_test.map(jump_lookup)

        # note that the last row of X_train contains information about the first entry in
        # y_test
        # we remedy this by removing this last row and making it the new first row of the test set
        X_test = pd.concat([X_train.iloc[-1:], X_test], ignore_index=True)
        X_train = X_train.iloc[:-1]
        y_test = pd.concat([y_train.iloc[-1:], y_test], ignore_index=True)
        y_train = y_train.iloc[:-1]

        trades = np.zeros(X_test.shape[0])
        capital = np.zeros(X_test.shape[0])
        current_capital = self.starting_capital

        print('---------------- Starting backtest ----------------')
        for idx in range(len(trades)):
            try:
                if idx == 0:
                    temp_model = self.train(X_train, y_train)
                elif idx % train_frequency == 0 and retrain:
                    temp_model = self.train(pd.concat([X_train, X_test.iloc[:idx]], ignore_index=True),
                                            pd.concat([y_train, y_test.iloc[:idx]], ignore_index=True))
                
                ###### BELOW IS FOR XGBOOST ONLY!!!! WILL NEED TO FIX TO SUPPORT OTHER MODELS
                next_prediction = temp_model.predict(xgb.DMatrix(X_test.iloc[[idx]], label=y_test.iloc[[idx]]))[0]
            except:
                print(f'Error in training/predicting at index {idx} of {len(trades)}')
                raise ValueError

            trades[idx] = strategy(next_prediction, X_test.iloc[[idx]]['close'].values[0], np.sum(trades), current_capital)
            cost_of_trade = trades[idx] * X_test.iloc[[idx]]['close'].values[0]
            current_capital -= cost_of_trade + self.commission * np.abs(cost_of_trade)
            capital[idx]= current_capital

        print('\n\n---------------- Backtest Complete! ----------------')

        results = pd.DataFrame({
            'open': X_test['open'],
            'high': X_test['high'],
            'low': X_test['low'],
            'close': X_test['close'],
            'position': trades,
            'capital': capital
        })

        return results