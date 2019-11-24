import tensorflow as tf
import numpy as np

from core.layers.model_ICNN_three import model_ICNN
from core.optimization.bundle_entropy import BundleEntropyMethod
from core.optimization.projected_newton import ProjNewtonLogistic
import core.optimization.pdipm as pdipm
from core.utils.utils import plot_function


G = np.array(
    [
        [-40.266327, 38.992367],
        [-37.990784, -16.280014],
        [-37.990784, -16.280014],
        [-41.761642, -13.359656],
    ]
)
h = np.array([88.05084, 101.968185, 103.24681, 112.075455])
pdipm.pdipm_boyd(G, h)

negQ = model_ICNN(
    [50, 50],
    [50, 50, 1],
    weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.3),
)

for i in range(10):
    print("we are in try {}".format(i))
    x = np.random.normal(10, 10, size=(4, 1)).astype(np.float32)
    y = np.random.normal(-0.5, 0.25, size=(1, 2, 1)).astype(np.float32)

    print("result", BundleEntropyMethod(negQ, x, y, 5, solver=pdipm.pdipm_boyd))

plot_function(negQ, x, regularized=True)

