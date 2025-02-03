"""
Use this class to keep track of session info and ultimately log backtest results
with all key info
"""
import numpy as np
from datetime import datetime
from pathlib import Path
from textwrap import fill
import os

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

    def add_backtest(self, backtest_results):
        self.backtest_results = backtest_results

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
        os.makedirs(path_to_output, exist_ok=True)

        time = datetime.now().strftime('%H:%M:%S')
        filename = self.strategy + time + '.txt'
        results_path = os.path.join(path_to_output, filename)

        with open(results_path, 'w') as results:
            results.write(self._get_session_info_string())


