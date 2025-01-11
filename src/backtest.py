import pandas as pd
import numpy as np

from typing import Callable, Any

def get_jump_lookup(num_classes):
    if num_classes == 3:
        jump_lookup = {
            'down':0,
            'neutral':1,
            'up':2
        }
        return jump_lookup
    elif num_classes == 5:
        jump_lookup = {
            'big_down':0,
            'small_down':1,
            'neutral':2,
            'small_up':3,
            'big_up':4
        }
    try:
        return jump_lookup
    except:
        print(f'Could not find a lookup for {num_classes} classes.')
        raise SystemError
    

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

    def run_backtest(self, strategy: Callable[[int, float, int, float], int]):
        """
        Run a backtest using the provided variables when the class was declared and according to
        a provided strategy. `strategies.py` has some default strategies, otherwise they can be
        declared and passed in.

        The `strategy` method must take 4 parameters and return an integer:

        params:
            int: `next_prediction` - an integer representing the prediction of the next price movement
                as defined by the appropriate jump lookup from `get_jump_lookup`.
            
            float: `price` - a float with the current price, typically the last closing price
            
            int: `current_pos` - an integer representing the current position on the coin

            float: `current_capital` - a float representing the current capital. It is up to the 
                `strategy` method to decide whether purchasing a coin is appropriate/possible
                given the current capital.
        
        return:
            int: an integer representing the change in position. For example, `+1` would be to buy
                1 coin, `-3` would be to sell 3 coins, and `0` is to do nothing.
        """
        # note that the last row of X_train contains information about the first entry in
        # y_test
        # we remedy this by removing this last row and making it the new first row of the test set
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        # make sure the target is of the following form
        jump_lookup = get_jump_lookup(self.num_classes)
        y_train = y_train.map(jump_lookup)
        y_test = y_test.map(jump_lookup)

        X_test = pd.concat([X_train.iloc[-1:], X_test], ignore_index=True)
        X_train = X_train.iloc[:-1]

        trades = np.zeros(X_test.shape[0])

        for idx in range(len(trades)):
            pass

    
class Strategy():
    def __init__(self, num_classes):
        self.num_classes = num_classes



if __name__ == '__main__':
    test = Backtest()