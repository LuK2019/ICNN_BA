import numpy as np


class GreedyEstimator:
    def __init__(self, stop_exploring_at, final_exploration_rate, stagnate_epsilon_at):
        """Controls the rate of exploration, i.e. the epsilon value of the bernouilli distribution.
        Idea: epsilon==1 for the stop_exploring_at % first episodes,
        then linear decline till stagnate_epsilon_at % of episodes at final_exploration_rate

        Args:
            stop_exploring_at: Till 0.1 * total_num_epsiodes, epsilon == 1
            final_exploration_rate: When decline has finished, this is the final epsilon
            stagnate_epsilon_at: 
        """
        self.stop_exploring_at = stop_exploring_at
        self.final_exploration_rate = final_exploration_rate
        self.stagnate_epsilon_at = stagnate_epsilon_at

    def get_epsilon(self, total_num_episode, current_episode):
        """ Return the current epsilon value for exploration.

        Args:
            total_num_episode: Total number of episodes of the simulation
            current_epsiode: Current epsiode of the simulation

        Returns:
            epsilon
        """
        if current_episode < total_num_episode * self.stop_exploring_at:
            return 1.0
        total_num_episode_to_decrease = (
            self.stagnate_epsilon_at * total_num_episode
            - total_num_episode * self.stop_exploring_at
        )
        if current_episode < (self.stagnate_epsilon_at * total_num_episode):
            increment_of_decrease_per_episode = (
                1 - self.final_exploration_rate
            ) / total_num_episode_to_decrease
            return 1.0 - increment_of_decrease_per_episode * (
                current_episode - total_num_episode * self.stop_exploring_at
            )
        else:
            return self.final_exploration_rate


if __name__ == "__main__":
    TOTAL = 100
    eps = GreedyEstimator(0.1, 0.1, 0.5)
    for i in range(TOTAL):
        print("Iteration {}: Epsilon = {}".format(i, eps.get_epsilon(TOTAL, i)))
