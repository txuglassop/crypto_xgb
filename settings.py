"""
configure settings for `__main__.py` here, including data to be used, test size
jump categories, feature engineering processes and some computing settings.
"""

import sys
from pathlib import Path
src_path = Path("src")
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

import src.eval_metrics as eval_metrics
import src.strategies as strategies

# data info
# data should be a string of the EXACT name of the desire csv file in  input/
# test size should be a float between 0 and 1 non inclusive
data_info = {
    'data': "BTCUSDT_1h_2020_2024_final.csv",
    'test_size': 0.20
}

# feature engineering
# num_classes is the number of different categories
# further dictionaries within this specify the margin,
# where they are ordered from lowest to highest (in terms)
# of category
feature_settings = {
    'num_classes': 3,

    'margins': {
        3: [0.005, 0.008], # down_margin, up_margin

        5: [0.008, 0.005, 0.005, 0.008] # big_down, small_down, small_up, big_up
    },

    'lag_factor': 5
}

train_test_settings = {
    'eval_metric': eval_metrics.f1_weighted_eval,

    'strategy':strategies.all_in_3_class,

    'starting_capital': 500_000,

    'commission': 0.002,

    'retrain': True,

    'train_frequency': 24*21
}