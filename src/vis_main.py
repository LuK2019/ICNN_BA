import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf

from core.utils.utils import model_loader, PlotFunction
from core.layers.modelpicnntwo import ModelPICNNTwo

matplotlib.rcParams["figure.figsize"] = [20, 10]


# Import Model
LOG_NUM = 1000
PATH_TO_MODEL = r"C:\Users\lukas\OneDrive\Universität\Mathematik\Bachelorarbeit\log_dir\log_{}\WEIGHTS\1".format(
    LOG_NUM
)


negQ = ModelPICNNTwo(
    [200],
    [200, 1],
    weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.4),
    name="negQ",
)

loaded_model = model_loader(negQ, PATH_TO_MODEL)
x = np.array([[5.0], [10.0], [1.0], [0.0]], dtype=np.float32)
optimum_arg = tf.Variable([[[0.5], [0.5]]])
PlotFunction(
    f=loaded_model, x=x, GRANULARITY=0.25, optimum_arg=optimum_arg, regularized=True
)

