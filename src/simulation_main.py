import tensorflow as tf
import numpy as np

from core.simulation.simulation import simulation
from core.simulation.game import game
from core.simulation.reward import RewardId
from core.simulation.random_generator import random_generator_uniform
from core.layers.model_ICNN import model_ICNN

from core.simulation.simulation import H


random_generator_uniform = random_generator_uniform(0.9, 1.1)
negQ = model_ICNN([100, 100], [100,100,1], name="negQ")
game = game(x_0=10., y_0=100.,S_0=10., T=2, alpha=0.01, random_generator=random_generator_uniform,reward_func=RewardId)
simulation = simulation(ICNN_model=negQ, game=game, num_episodes=5, size_minibatches=3, capacity_replay_memory=100, optimization_iterations=2)

x1 = tf.Variable([[3.],[2.],[1.]])
y1 = tf.Variable([[1.]]) 

x2 = tf.Variable([[4.],[1.],[1.]])
y2 = tf.Variable([[1.]]) 

test_vars = [[x1, y1], [x2, y2]]

negQ(test_vars[0])


# simulation.run_simulation()



























# # test the optimization on a model
# model = model_ICNN([100,200], [200,200,1])

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
