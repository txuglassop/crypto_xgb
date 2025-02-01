import pandas as pd
import numpy as np
import sys

def check_data(path: str):
    """
    runs some checks on the data given by `path`. this includes
    checking if the `timestamp` feature is ordered, the increments
    within the `timestamp` feature and their frequency

    params:
        path (str): path to the file (data)
    """
    try: 
        df = pd.read_csv(path)
    except:
        raise ValueError(f'Cannot find {path}')

    print(f'\n>>> Running data check on {path} <<<')

    l = df['timestamp']
    if all(l[i] <= l[i+1] for i in range(len(l) - 1)):
        print('Data is sorted')
    else:
        print('Data is NOT sorted - please check timestamp feature')
    
    incr = np.diff(l)
    unique, counts = np.unique(incr, return_counts=True)

    for u,c in zip(unique, counts):
        print(f'Incr: {u}, Count: {c}')

if __name__ == '__main__':
    path = sys.argv[1]
    check_data(path)