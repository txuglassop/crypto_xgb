

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