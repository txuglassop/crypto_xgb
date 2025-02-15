import numpy as np
import pandas as pd

close = np.random.rand(20)
window=2

print(pd.Series(close))

close_diffs = pd.Series(close).diff()

m1 = np.where(close_diffs >= 0, close_diffs, 0.0)
m2 = np.where(close_diffs >= 0, 0.0, -close_diffs)

print(m1, '\n\n', m2)

sm1 = pd.Series(m1).rolling(window).sum()
sm2 = pd.Series(m2).rolling(window).sum()

print(sm1)

print(sm2)

cmo = 100 * (sm1-sm2) / (sm1+sm2)

print(pd.Series(cmo))
