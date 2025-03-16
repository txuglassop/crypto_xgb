import pandas as pd
import numpy as np
import sys
import time

def check_data(df: pd.DataFrame):
    """
    runs some checks on the data given by `path`. this includes
    checking if the `timestamp` feature is ordered, the increments
    within the `timestamp` feature and their frequency, any duplicate
    rows, time since last update

    params:
        path (str): path to the file (data)
    """

    print(f'\n>>> Running data check <<<')

    l = df['timestamp']
    if all(l.iloc[i] <= l.iloc[i+1] for i in range(len(l) - 1)):
        print('Data is sorted\n')
    else:
        print('Data is NOT sorted - please check timestamp feature')

    last_entry_point = df['timestamp'].iloc[-1]
    time_since = time.time()*1000 - last_entry_point
    seconds = time_since // 1000
    minutes = seconds // 60
    hours = minutes // 60
    seconds %= 60
    minutes %= 60

    print('Time since last entry:')
    print(f'{hours} hours, {minutes} minutes, {seconds} seconds\n')

    duplicates = df.duplicated()
    num_dup = np.sum(duplicates)
    print(f'There are {num_dup} duplicate rows\n')

    incr = np.diff(l)
    unique, counts = np.unique(incr, return_counts=True)

    for u,c in zip(unique, counts):
        print(f'Incr: {u}, Count: {c}')


if __name__ == '__main__':
    path = sys.argv[1]
    df = pd.read_csv(path)
    check_data(df)
