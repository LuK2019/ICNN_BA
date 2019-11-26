import numpy as np


def Phi(delta_x, alpha):
    if alpha == 0:
        return delta_x
    else:
        return (np.exp(alpha * delta_x) - 1) / alpha


class Game:
    def __init__(self, x_0, y_0, S_0, T, alpha, random_generator, reward_func):
        """
        Args:
            x_0 = initial amount of stocks
            y_0 = initial amount of cash
            S_0 = initial stock price
            T: Number of periods till termination, i.e. we liquidate our stocks at t=T-1
            alpha: liquidity parameter
            random_generator: instance of random_generator object
        """
        for element in [x_0, y_0, S_0]:
            assert isinstance(
                element, float
            ), "The initial values have to float dtype, the element {} is type{} ".format(
                element, type(element)
            )
        self.x_0 = x_0
        self.y_0 = y_0
        self.S_0 = S_0
        self.alpha = alpha
        self.random_generator = random_generator
        self.T = T
        self.reward_func = reward_func

    def get_new_state(
        self, current_state, action, action_decimals=3, printout: bool = False
    ) -> dict:
        """ Return the transition tuple for a given current_state, action
            Conventions:
            "current_state": np.ndarray [4,1], [[num_stocks (x)], [amout_cash (y)], [current_price (S)], [num_period (t)]]
            "action": np.ndarray [2,1]

                Returns: ( 
                {
                "current_state": np.ndarray [4,1],
                "action": np.ndarray [2,1]
                "reward": float/np.ndarray [1,] 
                "next_state": np.ndarray [4,1]
                }:dict,
                liqudity cost:float)

            Args:
                printout:bool: Decide if you want to printout current choices and their results
            """

        # Unpack the current_state
        x, y, S, t = current_state[:, 0]
        # If necessary, convert action to np.ndarray
        if not isinstance(action, np.ndarray):
            action = action.numpy()
        # Round action to decimal threshold to improve numerical stability
        action = np.round(action, decimals=action_decimals)

        # Check validity of the actions provided
        assert (
            (0 <= action[0][0])
            & (action[0][0] <= 1)
            & (0 <= action[1][0])
            & (action[1][0] <= 1)
        ), "Invalid action taken, must be within [0,1], action={}".format(action)

        # Check validity of the state
        assert (
            y >= 0
        ), "The current state has a negative cash balance, expected it to be non-negative, current state: {}, action {}".format(
            current_state, action
        )

        # Calculate the transition for periods before the final period
        if t < self.T - 1:
            # 1. a0 inference:
            num_stocks_to_sell = x * action[0][0]
            cash_from_selling_stocks = num_stocks_to_sell * S

            # 2. a1 inference:
            amount_cash_to_invest = y * action[1][0]

            # 3. change in cash desired (1. + 2.)
            change_in_cash_desired = cash_from_selling_stocks - amount_cash_to_invest

            # 4. upper and lower bound for the change in cash desired
            # It needs to hold: 0 < y + change_in_cash_desired < y + (change in cash via selling all stocks with liquidity costs)
            # 4.1. lower bound
            lower_bound = 0
            # 4.2. upper bound
            upper_bound = y - Phi(-x, self.alpha) * S

            # The agent does not know about liquidity effects
            required_change_in_stocks_for_change_in_cash = -change_in_cash_desired / S

            # Check bounds
            next_y_before_check = (
                y - Phi(required_change_in_stocks_for_change_in_cash, self.alpha) * S
            )
            # Case 1: y+change_in_cash_desired < 0
            if next_y_before_check < lower_bound:
                print(
                    "Next_y_before_check: {} < 0, expected it to be non-negative".format(
                        next_y_before_check
                    )
                )
                required_change_in_stocks_for_change_in_cash = (
                    1 / self.alpha
                ) * np.log(y * self.alpha * (1 / S) + 1)
            # Case 2: change_in_cash_desired > (change in cash via selling all stocks with liquidity costs)
            if next_y_before_check > upper_bound:
                required_change_in_stocks_for_change_in_cash = -x

            # Check validity of calculations
            if (change_in_cash_desired > 0) & (
                required_change_in_stocks_for_change_in_cash > 0
            ):
                print(
                    "LOGICAL ERROR, change_in_cash_desired {} is positive but the required_change_in stocks is positive as well {})".format(
                        change_in_cash_desired,
                        required_change_in_stocks_for_change_in_cash,
                    )
                )

            # 5. next_x
            next_x = x + required_change_in_stocks_for_change_in_cash

            assert next_x >= 0, "We have negative cash balance! {}".format(next_x)
            assert (
                next_x <= x + y / S
            ), " We did lend money to buy more stocks! {}".format(next_x)

            delta_x = next_x - x

            next_y = y - Phi(delta_x, self.alpha) * S
            if np.abs(next_y) < 0.0001:
                next_y = 0.0

            next_S = S * self.random_generator.generate()
            next_t = t + 1

            next_state = np.array(
                [[next_x], [next_y], [next_S], [next_t]], dtype=np.float32
            )

            change_in_cash = next_y - y
            if printout:
                print(
                    "You bought/sold {} stocks for which you paid/received on average {}, that is {} less than for current price {} and spent in total {}".format(
                        delta_x,
                        (next_y - y) / delta_x,
                        (next_y - y) / delta_x - S,
                        S,
                        next_y - y,
                    )
                )
            return_dict = {
                "current_state": current_state,
                "action": action,
                "reward": self.reward_func(change_in_cash),
                "next_state": next_state,
            }

        # Calculate transition to the final period, here all stocks have to be liquidated
        else:
            next_x = 0
            delta_x = next_x - x
            next_y = y - Phi(delta_x, self.alpha) * S
            change_in_cash = next_y - y
            next_S = S * self.random_generator.generate()
            next_t = t + 1
            next_state = np.array(
                [[next_x], [next_y], [next_S], [next_t]], dtype=np.float32
            )
            return_dict = {
                "current_state": current_state,
                "action": action,
                "reward": self.reward_func(change_in_cash),
                "next_state": next_state,
            }

        # Check validity of the next state
        assert (
            return_dict["next_state"][1, 0] >= 0
        ), "The next state has negative cash balance, expected it to be non negative. Transition: Current state {}, next state {}, action {}".format(
            return_dict["current_state"],
            return_dict["next_state"],
            return_dict["action"],
        )

        # Calculate liquidity costs, i.e. the difference in cash after the trade between the scenarios with and without liquidity costs
        # denote the amount of cash with liquidity costs y_1_L, without liquidity costs y_1:
        # 0 \leq y_1_L - y_1 = (delta_x-Phi(delta_x))S_0
        actual_delta_x = (
            return_dict["next_state"][0, 0] - return_dict["current_state"][0, 0]
        )
        liquidity_costs = (
            actual_delta_x - Phi(actual_delta_x, self.alpha)
        ) * return_dict["current_state"][2, 0]
        return (return_dict, liquidity_costs)


if __name__ == "__main__":
    from random_generator import RandomGeneratorUniform
    from reward import RewardId

    random_generator = RandomGeneratorUniform(0.9, 1.1)
    game = Game(
        x_0=1.0,
        y_0=1.0,
        S_0=1.0,
        T=5,
        alpha=0.1,
        reward_func=RewardId,
        random_generator=random_generator,
    )

