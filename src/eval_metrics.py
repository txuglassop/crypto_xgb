"""
NOTE: each eval metric method MUST return
```
<str>, <float>
```
where <str> MUST be the `.__name__` property, i.e. the name of the 
module. `True` and `False` can also be added to indicate high/lower 
score is better
"""

import numpy as np
import xgboost as xgb

from sklearn.metrics import f1_score


def f1_weighted_eval(predt: np.ndarray, dtrain: xgb.DMatrix):
    y_true = dtrain.get_label()
    y_pred = np.argmax(predt, axis=1)
    
    f1 = f1_score(y_true, y_pred, average='macro')
    return 'f1_weighted_eval', f1