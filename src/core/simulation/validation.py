import numpy as np 

def gamma(current_state:"np.array", game:"game") -> "float":
    """Calculates the upper threshold of stocks we are able to buy 
    given the amout of money y available.
    Args:
        current_state: np.array [4,1]
        game: game object
    Returns:
        float
    """
    x,y,S,t = current_state[:,0]
    return np.log((y*game.alpha)/(S) +1)*(1/game.alpha) + x
    


def optimum_2p_solution(current_state:"np.array", game:"game") -> "float":
    """ Calculates the optimum desired amount of stocks in period 1
    to maximize expected cash value at liquidation.
    
    Args:
        current_state: np.array [4,1]
        game: game object
    Returns:
        float
    """
    x,y,S,t = current_state[:,0]
    mu = game.random_generator.mean
    x_star = (np.log(mu)+x*game.alpha)/(2*game.alpha)
    gamma_val = gamma(current_state, game)
    if x_star < 0:
        return 0
    if (0 <= x_star) & (x <= gamma_val):
        return x_star
    else:
        return gamma_val



