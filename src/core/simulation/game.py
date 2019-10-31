import numpy as np 

def phi(delta_x, alpha):
    if alpha == 0: 
        return delta_x
    else:
        return (np.exp(alpha*delta_x)-1)/alpha

class game:
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
            assert isinstance(element, float), "The initial values have to float dtype, the element {} is different".format(element)
        self.x_0 = x_0
        self.y_0 = y_0
        self.S_0 = S_0
        self.alpha = alpha
        self.random_generator = random_generator
        self.T = T
        self.reward_func=reward_func

    def get_new_state(self, current_state, action):
        """Return the  tuple
        Args:
            current_state = tuple like (num_stocks (x), amout_cash(y), current_price(S), num_period(t))
            action = tuple like (action[0], action[1]); action[0]: %-of stocks to sell, action[1]: %-of cash to invest
        
        Returns:
            dict of (current_state, action, reward, next_state)
        """
        x,y,S,t = current_state
               
        assert (0 <= action[0]) & (action[0] <= 1) & (0 <= action[1]) & (action[1] <= 1), "Invalid action taken, must be within [0,1], action={}".format(action)
        if not isinstance(action, np.ndarray):
            action = action.numpy()
            action = (action[0][0], action[1][0])
    
        if t < self.T-1:
            num_stocks_to_sell = x*action[0]
            cash_from_selling_stocks = x*action[0]*S

            amount_cash_to_invest = y*action[1]

            change_in_cash_desired = cash_from_selling_stocks - amount_cash_to_invest

            required_change_in_stocks_for_change_in_cash = -change_in_cash_desired/S # Here we see that the agent does not know about the liquidity effects

            if (change_in_cash_desired > 0) & (required_change_in_stocks_for_change_in_cash > 0):
                print("LOGICAL ERROR, change_in_cash_desired {} is positive but the required_change_in stocks is positive as well {})".format(change_in_cash_desired, required_change_in_stocks_for_change_in_cash))

            print("change_in_cash_desired", change_in_cash_desired)
            print("Required_change_in_stocks_for_change_in_cash", required_change_in_stocks_for_change_in_cash)

            next_x = x + required_change_in_stocks_for_change_in_cash

            assert next_x >= 0, "We have negative cash balance! {}".format(next_x)
            assert next_x <= x + y/S, " We did lend money to buy more stocks! {}".format(next_x)

            delta_x = next_x - x

            next_y = y - phi(delta_x, self.alpha)*S

            next_S = S * self.random_generator.generate()
            next_t = t + 1

            next_state = (next_x, next_y, next_S, next_t)

            change_in_cash = next_y - y
            return {"current_state": current_state, "action":action, "reward":self.reward_func(change_in_cash), "next_state":next_state}
        #TODO: Implement the ending case

if __name__ == "__main__":
    from random_generator import random_generator_uniform
    from reward import RewardId

    random_generator = random_generator_uniform(0.9, 1.1)
    game = game(T=5, alpha=0.1,reward_func=RewardId, random_generator=random_generator)

    print(game.get_new_state((10, 100, 10, 0), (1., 0.)))
    for index, transition in replay_memory.iterrows():
        print(transition)