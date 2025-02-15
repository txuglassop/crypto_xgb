"""
Use this class to keep track of session info and ultimately log backtest results
with all key info
"""
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from textwrap import fill
import os

from backtest_metrics import get_backtest_metrics_string
from sklearn.metrics import classification_report, confusion_matrix

class SessionInfo():
    def __init__(
            self,
            data: str,
            num_classes: int,
            lag_factor: int,
            test_size: float,
            up_margin = None,
            down_margin = None,
            big_down_margin = None,
            small_down_margin = None,
            small_up_margin = None,
            big_up_margin = None,
    ):
        self.data = data
        self.num_classes = num_classes
        self.lag_factor = lag_factor
        self.test_size = test_size

        self.up_margin = up_margin
        self.down_margin = down_margin
        
        self.big_down_margin = big_down_margin
        self.small_down_margin = small_down_margin
        self.small_up_margin = small_up_margin
        self.big_up_margin = big_up_margin

    def add_features(self, features: list):
        self.features = features
    
    def add_strategy(self, strategy: str):
        self.strategy = strategy

    def add_backtest(self, backtest_results: pd.DataFrame):
        self.backtest_results = backtest_results
        self.backtest_results.__name__ = self.strategy

    def _get_session_info_string(self):
        features = fill(", ".join(self.features), width=50, subsequent_indent="")
        string = f"""
================== Session Info ==================

Data:                           {self.data}
Num Classes:                    {self.num_classes}
Lag Factor:                     {self.lag_factor}
Test Size:                      {self.test_size}

Up Margin:                      {self.up_margin}
Down Margin:                    {self.down_margin}


-------------------- Features --------------------

{features}

"""
        return string

    def log_session(self, path_to_output: str):
        """
        logs all session information, predictors, and back test results in a directory
        within the specified path

        params:
            `path_to_output` (str) - a path to where the session is to be logged
        """
        # make new directory
        time = datetime.now().strftime('%d-%m-%y_%H:%M:%S')
        dirname = path_to_output + self.data.split('_')[0] + '_' + time
        os.makedirs(dirname, exist_ok=True)

        filename = 'summary.txt'
        summary_path = os.path.join(dirname, filename)

        class_report_string = f'''
-------------- Classification Report --------------
{classification_report(self.backtest_results['actual'], self.backtest_results['predictions'])}

---------------- Confusion Matrix ----------------
{confusion_matrix(self.backtest_results['actual'], self.backtest_results['predictions'])}
'''

        with open(summary_path, 'w') as f:
            f.write(self._get_session_info_string())
            f.write(get_backtest_metrics_string(self.backtest_results))
            f.write(class_report_string)

        self.backtest_results.to_csv(dirname + '/backtest.csv')

        

            


