# Standard packages
import tensorflow as tf
import numpy as np
import time

# Custom packages
from core.simulation.simulation import Simulation
from core.simulation.game import Game
from core.simulation.reward import RewardId
from core.simulation.random_generator import RandomGeneratorUniform
from core.layers.modelpicnnthree import ModelPICNNThree
from core.layers.modelpicnntwo import ModelPICNNTwo
from core.simulation.validation import Optimum2PeriodSolution
import matplotlib.pyplot as plt

from core.simulation.greedy_estimator import GreedyEstimator

# Initalize necessary simulation objects:

# Define the random generator for the price process
random_generator_uniform = RandomGeneratorUniform(0.8, 1.8)

# Define the neural network
# 2 Layer PICNN
negQ = ModelPICNNTwo(
    [200],
    [200, 1],
    weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.4),
    name="negQ",
)

# Define the game setting
# Two period game
game_two_period = Game(
    x_0=5.0,
    y_0=10.0,
    S_0=1.0,
    T=2,
    alpha=1,
    random_generator=random_generator_uniform,
    reward_func=RewardId,
)

# Three period game
game_three_period = Game(
    x_0=1.0,
    y_0=10.0,
    S_0=1.0,
    T=3,
    alpha=0.01,
    random_generator=random_generator_uniform,
    reward_func=RewardId,
)
# Define exploration process via the epsilon greedy factor
greedy_estimator = GreedyEstimator(
    stop_exploring_at=0.075, final_exploration_rate=0.1, stagnate_epsilon_at=0.25
)

# Setup the simulation object
simulation = Simulation(
    greedy_estimator=greedy_estimator,
    ICNN_model=negQ,
    game=game_two_period,
    num_episodes=50000,
    ITERATIONS=1,
    size_minibatches=1,
    capacity_replay_memory=1,
    optimization_iterations=3,
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.000025),
    discount_factor=0.5,
    show_plot_every=1000000,
    LOG_NUM=3336,
    initial_action_for_optimization=tf.Variable([[0.7], [0.2]]),
)

print(
    "\n==============",
    "The optimal choice of this simulation is {}, with expected value of the random generator {}".format(
        Optimum2PeriodSolution(
            np.array(
                [
                    [game_three_period.x_0],
                    [game_three_period.y_0],
                    [game_three_period.S_0],
                    [0],
                ]
            ),
            game_three_period,
        ),
        random_generator_uniform.mean,
    ),
)

# Run the simulation with the specified parameters
start_simulation = time.time()

simulation.run_simulation()

end_simulation = time.time()

print("Simulation took {}".format(end_simulation - start_simulation))

print("First Weight:", simulation.negQ.trainable_variables[0])
