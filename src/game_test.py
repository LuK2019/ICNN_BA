# Standard packages
import tensorflow as tf
import numpy as np
import time

# Custom packages
from core.simulation.simulation import simulation
from core.simulation.game import game
from core.simulation.reward import RewardId
from core.simulation.random_generator import random_generator_uniform
from core.layers.model_ICNN_three import model_ICNN_three
from core.layers.model_ICNN_two import model_ICNN_two
from core.simulation.validation import optimum_2p_solution
import matplotlib.pyplot as plt
from core.simulation.simulation import H

random_generator_uniform = random_generator_uniform(0.7, 1.3)


game_two_period = game(
    x_0=1.0,
    y_0=10.0,
    S_0=1.0,
    T=2,
    alpha=0.01,
    random_generator=random_generator_uniform,
    reward_func=RewardId,
)

game_three_period = game(
    x_0=1.0,
    y_0=10.0,
    S_0=1.0,
    T=3,
    alpha=0.01,
    random_generator=random_generator_uniform,
    reward_func=RewardId,
)


current_state = np.array([[6.97], [3.8], [0.74], [1.0]])
action = np.array([[1], [0.0001]])
action_smaller = action * 0.00000001

game_three_period.get_new_state(current_state, action_smaller)

threshold = 0.001
action
