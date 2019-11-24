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

# For the 3 layer PICNN case:
# negQ = model_ICNN_three (
#     [50, 50],
#     [50, 50, 1],
#     activation_func="relu",
#     weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5),
#     name="negQ",
# )

# Initalize necessary simulation objects:
random_generator_uniform = random_generator_uniform(0.7, 1.3)

# For the 2 layer PICNN case:
negQ = model_ICNN_two(
    [200],
    [200, 1],
    weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.4),
    name="negQ",
)

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


simulation = simulation(
    ICNN_model=negQ,
    game=game_three_period,
    num_episodes=10000,
    ITERATIONS=1,
    size_minibatches=1,
    capacity_replay_memory=1,
    optimization_iterations=3,
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.000025),
    discount_factor=0.5,
    show_plot_every=9999,
)

print(
    "\n==============",
    "The optimal choice of this simulation is {}, with expected value of the random generator {}".format(
        optimum_2p_solution(
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

# Inspect the simulation results
df = simulation.simulation_summary
print(df)

df.to_pickle(r"C:\Users\lukas\Desktop\01_25_k.pkl")

df.plot(x="episode", y="final cash balance")
plt.show()
df.plot(x="episode", y="deviation of x_1 from optimal x_1")
plt.show()

print(simulation.replay_memory.iloc[:6, :])
print(simulation.replay_memory.tail())


# # test the optimization on a model
# model = model_ICNN_three([100,200], [200,200,1])

# # Data
# x = tf.Variable([[4.],[2.],[1.]], dtype="float32")
# y = tf.Variable([[4.]], dtype="float32")

# # Prediction
# pred = model((x,y))
# print("Forward Pass", pred)

# # Optimization
# result = BundleEntropyMethod(model, x, y, 10)

# print("The result of the optimization", result)

# args = np.linspace(0,1,num=100)
# pred_list = [model((x,tf.Variable([[arg]], dtype="float32"))) for arg in args]
# pred_list_np = [i.numpy() for i in pred_list]
# pred_list_np_zoom = [i[0][0]for i in pred_list_np]
# import matplotlib.pyplot as plt
# plt.scatter(args, pred_list_np_zoom)
# plt.show()


# tf.multiply(y, tf.matmul(self.W_0_yu, x) + self.b_0_y))

# W_0_yu = [v for v in model.trainable_variables if v.name == "model_icnn_2/z_1/W_0_yu:0"][0]
# b_0_y = [v for v in model.trainable_variables if v.name == "model_icnn_2/z_1/b_0_y:0"][0]

# right_side = tf.matmul(W_0_yu, x) + b_0_y
# var
# var = model.trainable_variables

# def f(arg):
#     x,y = arg
#     return 6*y[0]**2 + 6*y[1]**2

# BundleEntropyMethod(f, x, tf.Variable([[1.],[1.]], dtype="float32"),10)
