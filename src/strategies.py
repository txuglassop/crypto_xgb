"""
Generic strategy declaration:

def strategy(next_prediction: int, price: float, current_pos: int, current_capital: float) -> int:
    pass

A `strategy` method must take 4 parameters and return an integer:

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

def basic_3_class_strategy(next_prediction: int, price: float, current_pos: int, current_capital: float) -> int:
    """
    If predicted up, then `+1`, if predicted down, then sell everything, otherwise do nothing (`0`)
    """
    if next_prediction == 0:
        return -current_pos
    elif next_prediction == 2 and current_capital > price:
        return +1
    else:
        return 0