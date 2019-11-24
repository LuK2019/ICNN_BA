from core.optimization.bundle_entropy import BundleEntropyMethod
import tensorflow as tf
import numpy as np
from core.layers.model_ICNN_three  import model_ICNN_three 
from core.optimization.projected_newton import ProjNewtonLogistic


negQ = model_ICNN_three ([1000, 100], [1000, 100, 1])


A = np.array([[90.440674, 72.57225], [-3.1540527, 39.155697], [33.440674, 22.57225]])
b = np.array([90.1675, 90.4501, 90.4501])

factor = np.max(b) ** (-1)
A_ = factor * A
b_ = factor * b

# print(ProjNewtonLogistic(A, b, factor=1))

print(ProjNewtonLogistic(A_, b_, factor=factor))

