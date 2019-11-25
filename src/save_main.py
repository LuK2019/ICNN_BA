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

negQ1 = model_ICNN_two(
    [200],
    [200, 1],
    weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.4),
    name="negQ",
)
negQ2 = model_ICNN_two(
    [200],
    [200, 1],
    weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.4),
    name="negQ",
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


simulation1 = simulation(
    ICNN_model=negQ1,
    game=game_three_period,
    num_episodes=10,
    ITERATIONS=1,
    size_minibatches=1,
    capacity_replay_memory=1,
    optimization_iterations=3,
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.000025),
    discount_factor=0.5,
    show_plot_every=9999,
)
simulation2 = simulation(
    ICNN_model=negQ2,
    game=game_three_period,
    num_episodes=10,
    ITERATIONS=1,
    size_minibatches=1,
    capacity_replay_memory=1,
    optimization_iterations=3,
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.000025),
    discount_factor=0.5,
    show_plot_every=9999,
)


simulation1.run_simulation()

model1 = simulation1.negQ

simulation2.run_simulation()

model2 = simulation2.negQ


print(model1.trainable_variables[0][:5])
print(model2.trainable_variables[0][:5])


model1.save_weights(r"C:\Users\lukas\Desktop\log_res\weightsss\22")
model2.load_weights(r"C:\Users\lukas\Desktop\log_res\22")


print(model1.trainable_variables[0][:5])
print(model2.trainable_variables[0][:5])

# ### CHECKPOINTS ####
# import os

# CHECKPOINT_DIR = r"C:\Users\lukas\Desktop\log_res\10"
# CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")

# checkpoint = tf.train.Checkpoint(model=model)

# checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

# status = checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))


# #### SERIALIZATION VIA SaveModel API ###
# SAVE_MODEL_PATH = r"C:\Users\lukas\Desktop\log_res\2"

# tf.saved_model.save(model, SAVE_MODEL_PATH)


# model_loaded = tf.saved_model.load(SAVE_MODEL_PATH)

# DEFAULT_FUNCTION_KEY = "serving_default"

# inference_func = model_loaded.signatures[DEFAULT_FUNCTION_KEY]
