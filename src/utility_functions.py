import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report

def classification_summary(y_pred: np.array, y_true: np.array):
    """
    Prints a summary of classification results.

    Params:
        np.array: y_pred - predictions on y_test
        np.array: y_true - actual test data, i.e. y_test

    returns:
        nothing 
    """
    print('\n-------------- Classification Report --------------')
    print(classification_report(y_true, y_pred))

    print('\n\n---------------- Confusion Matrix ----------------')
    print(confusion_matrix(y_true, y_pred))

def get_jump_lookup(num_classes: int) -> dict:
    """
    Returns a lookup table that converts string categories for given
    `num_classes` (from using the `add_jump_category` functions in 
    `feature_engineering.py`) to integers.
    """
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
    
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def print_progress_bar (iteration, total, prefix = 'Progress:', suffix = 'Complete', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def get_monthly_returns(timestamp: pd.Series, equity: pd.Series) -> list:
    """
    Returns an array of monthly returns for historical equity. Note the
    value of `timestamp` and `equity` at a given index refer to the same point
    in time

    params:
        timestamp (np.array) - UNIX timestamps in ms

        equity (np.array) - equity positions that correspond to `timestamp`

    returns:
        an array containing the monthly returns
    """
    assert len(timestamp) == len(equity)
    returns = list()

    months = pd.to_datetime(timestamp, unit='ms').dt.month
    cur_month = months[0]
    start = equity[0]

    for idx in range(len(months) - 1):
        if months[idx+1] == cur_month:
            continue # still in the same month
        end = equity[idx]
        returns.append(end / start - 1)
        start = equity[idx+1]
        cur_month = months[idx+1]
    returns.append(equity[len(equity)-1] / start - 1)

    return returns
