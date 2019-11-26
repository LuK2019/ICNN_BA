import numpy as np
from .game import Phi


def Gamma(current_state: "np.array", game: "game") -> "float":
    """Calculates the upper threshold of stocks we are able to buy 
    given the amout of money y available.
    Args:
        current_state: np.array [4,1]
        game: game object
    Returns:
        float
    """
    x, y, S, t = current_state[:, 0]
    return np.log((y * game.alpha) / (S) + 1) * (1 / game.alpha) + x


def Optimum2PeriodSolution(current_state: "np.array", game: "game") -> "float":
    """ Calculates the optimum desired amount of stocks in period 1
    to maximize expected cash value at liquidation.
    
    Args:
        current_state: np.array [4,1]
        game: game object
    Returns:
        float
    """
    x, y, S, t = current_state[:, 0]
    mu = game.random_generator.mean
    x_star = (np.log(mu) + x * game.alpha) / (2 * game.alpha)
    gamma_val = Gamma(current_state, game)
    if x_star < 0:
        return 0
    if (0 <= x_star) & (x_star <= gamma_val):
        return x_star
    else:
        return gamma_val


def FinalCashBalanceWithOptimalChoice(
    initial_state, first_state, optimal_choice_of_next_x1, game
):
    """ In period 1, calculate the final cash balance, based on the optimal choice of the next x1 and
    the price in period 1
    Args:
        initial_state: np.ndarray of the initial_state in period 0 
        first_state: np.ndarray of period 1
        optimal_choice_of_next_x1,: return value of the function Optimum2PeriodSolution
        game: game object to obtain the alpha
    Returns:
        final cash balance of the optimal choice
    """
    # Unpacking the initial and the first state
    initial_x, initial_y, initial_S, initial_t = initial_state[:, 0]
    first_x, first_y, first_S, first_t = first_state[:, 0]
    # Assert correctness of the states
    assert (
        initial_t == 0
    ), "Expected the initial state to be of period 0, initial state {}".format(
        initial_state
    )
    assert (
        first_t == 1
    ), "Expected the initial state to be of period 1, second state {}".format(
        first_state
    )
    # Calculate cash balance in  period 1 after the optimal choice of x1
    delta_x = optimal_choice_of_next_x1 - initial_x
    first_y = initial_y - Phi(delta_x, game.alpha) * initial_S
    # Clip to improve numerical stability
    if np.abs(first_y) < 0.0001:
        first_y = 0.0
    # Calculate stock balance in  period 1
    first_x = optimal_choice_of_next_x1
    # Ensure the validity of the cash balance
    assert first_y >= 0, "Cash balance is negative {}".format(first_y)

    # Calculate final cash balance
    final_x = 0  # Liquidate all cash
    delta_x = final_x - first_x
    final_y = first_y - Phi(delta_x, game.alpha) * first_S
    # Ensure the validity of the cash balance
    assert final_y >= 0, "Cash balance is negative {}".format(final_y)
    return final_y
