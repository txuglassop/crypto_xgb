"""
Generic strategy declaration:

def strategy(next_prediction: int, price: float, current_pos: float, current_capital: float, delta: float) -> int:
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
        
        float: `delta` - the delta of our trade i.e. the change in our position since we entered
            the trade - use this for take-profits and stop-losses.
        
    return:
        int: an integer representing the change in position. For example, `+1` would be to buy
            1 coin, `-3` would be to sell 3 coins, and `0` is to do nothing.    

"""
def all_in_3_class(next_prediction: int, price: float, current_pos: float, current_capital: float, delta: float) -> int:
    """
    If predicted up, then buy all possible, if predicted down, sell everything
    """
    if next_prediction == 0:
        return -current_pos
    elif next_prediction == 2:
        return current_capital // price
    else:
        return 0
    
def stop_loss_3_class(next_prediction: int, price: float, current_pos: float, current_capital: float, delta: float) -> int:
    """
    If predicted up, then buy all possible, if predicted down OR our delta has crossed over our stop loss, sell everything
    """
    stop_loss = -0.08 # -8% stop loss
    if next_prediction == 0 or delta < stop_loss:
        return -current_pos
    elif next_prediction == 2:
        return current_capital // price
    else:
        return 0
    
def take_profit_3_class(next_prediction: int, price: float, current_pos: float, current_capital: float, delta: float) -> int:
    """
    if delta has crossed over our take profit, sell, same with stop loss
    """
    take_profit = 0.11
    stop_loss = -0.08
    if next_prediction == 0 or delta < stop_loss or delta > take_profit:
        return -current_pos
    elif next_prediction == 2:
        return current_capital // price
    else:
        return 0

def short_3_class(next_prediction: int, price: float, current_pos: float, current_capital: float, delta: float) -> int:
    """
    If predicted down, go short, where we can by 80% of capital. If up, return to neutral.
    """
    if next_prediction == 2:
        return -current_pos
    elif next_prediction == 0 and current_pos == 0:
        return -0.8 * (current_capital / price)
    else:
        return 0
    
def L_S_5_class(next_prediction: int, price: float, current_pos: float, current_capital: float, delta: float) -> int:
    """
    If predicted big_up, go as long as possible. If predicted big_down, go as 80% short. Exit position
    if predicted in opposite direction
    """
    if next_prediction == 4 and current_pos <= 0:
        current_capital -= current_pos * price
        return -current_pos + 0.99 * (current_capital / price)
    elif next_prediction == 0 and current_pos >= 0:
        return -current_pos - 0.8 * (current_capital / price)
    elif (current_pos > 0 and next_prediction == 1) or (current_pos < 0 and next_prediction == 3):
        return -current_pos
    else:
        return 0