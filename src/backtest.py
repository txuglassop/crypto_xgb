import pandas as pd
import numpy as np
import xgboost as xgb

from typing import Callable, Any
from utility_functions import get_jump_lookup, print_progress_bar 

class Backtest():
    def __init__(
        self, 
        model_trainer: Callable[[np.ndarray, np.ndarray], Any], 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_test: np.ndarray,
        y_test: np.ndarray,
        train_timestamp: np.array,
        test_timestamp: np.array,
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

            train_timestamp, test_timestamp - timestamps corresponding to train and
                    test sets.

            int: num_classes - the number of classes in the target variable

            float: starting capital - your starting capital! used in backtests

            float: commission - the broker commission - used in backtests
        """
        assert X_train.shape[0] == len(train_timestamp)
        assert X_test.shape[0] == len(test_timestamp)

        self.train = model_trainer
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.train_timestamp = train_timestamp
        self.test_timestamp = test_timestamp
        self.num_classes = num_classes
        self.starting_capital = starting_capital
        self.commission = commission

    def run_backtest(self, strategy: Callable[[int, float, int, float, float], int],
                     retrain = True, train_frequency = 7, progress_bar = True):
        """
        Run a backtest using the provided variables when the class was declared and according to
        a provided strategy. `strategies.py` has some default strategies, otherwise they can be
        declared and passed in.

        params:
            strategy - as outlined in `strategies.py`

            test_timestamp - an array of timestamps that correspond to the observation in y_test

            bool: retrain - set to `True` if the model is to be retrained on test data
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

        test_timestamp = pd.concat([self.train_timestamp.iloc[-1:], self.test_timestamp], ignore_index=True)
        train_timestamp = self.train_timestamp.iloc[:-1]

        trades = np.zeros(X_test.shape[0])
        capital = np.zeros(X_test.shape[0])
        predictions = np.zeros(X_test.shape[0])
        current_capital = self.starting_capital
        delta = 0.0
        buy_price = 0.0

        print('-------------------- Starting backtest --------------------\n')

        for idx in range(len(trades)):
            try:
                if idx == 0:
                    temp_model = self.train(X_train, y_train)
                elif idx % train_frequency == 0 and retrain:
                    temp_model = self.train(pd.concat([X_train, X_test.iloc[:idx]], ignore_index=True),
                                            pd.concat([y_train, y_test.iloc[:idx]], ignore_index=True))

                ###### BELOW IS FOR XGBOOST ONLY!!!! WILL NEED TO FIX TO SUPPORT OTHER MODELS
                next_prediction = temp_model.predict(xgb.DMatrix(X_test.iloc[[idx]], label=y_test.iloc[[idx]]))[0]
                next_prediction = np.argmax(next_prediction)
            except:
                raise ValueError(f'Error in training/predicting at index {idx} of {len(trades)}')

            predictions[idx] = next_prediction
            cur_price = X_test.iloc[[idx]]['close'].values[0]

            # if we are currently long, find our current delta
            if np.sum(trades) > 0:
                delta = cur_price / buy_price - 1

            trade = strategy(next_prediction, cur_price, np.sum(trades), current_capital, delta)
            trades[idx] = trade

            # check if we made a change in our position
            if trade > 0:
                # we just entered a new long position - keep track of delta
                buy_price = cur_price
            elif trade < 0:
                # we have exited our position - our delta from here is 0
                delta = 0

            cost_of_trade = trades[idx] * cur_price
            current_capital -= cost_of_trade + self.commission * np.abs(cost_of_trade)
            capital[idx]= current_capital

            if progress_bar: print_progress_bar(idx+1, len(trades))

        print('\n\n-------------------- Backtest Complete! --------------------')

        results = pd.DataFrame({
            'timestamp': test_timestamp,
            'open': X_test['open'],
            'high': X_test['high'],
            'low': X_test['low'],
            'close': X_test['close'],
            'predictions': predictions,
            'actual': y_test,
            'trades': trades,
            'capital': capital
        })

        results['equity'] = results['capital'] + results['trades'].cumsum() * results['close']

        results.__name__ = strategy.__name__

        return results

if __name__ == '__main__':
    print('ok')