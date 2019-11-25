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
from core.utils.utils import model_loader
import matplotlib.pyplot as plt
from core.simulation.simulation import H


random_generator_uniform = random_generator_uniform(0.7, 1.3)

negQ1 = model_ICNN_two(
    [200],
    [200, 1],
    weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.4),
    name="negQ",
)

LOG_NUM = 777

PATH_TO_MODEL = r"C:\Users\lukas\OneDrive\Universität\Mathematik\Bachelorarbeit\log_dir\log_{}\WEIGHTS\1".format(
    LOG_NUM
)

loaded_model = model_loader(negQ1, PATH_TO_MODEL)

loaded_model.trainable_variables[0]
