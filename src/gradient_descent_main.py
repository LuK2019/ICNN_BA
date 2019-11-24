import tensorflow as tf
import numpy as np
from core.layers.model_ICNN_three  import model_ICNN
from core.optimization.gradient_descent import GradientDescent
from core.optimization.bundle_entropy import BundleEntropyMethod



model = model_ICNN([5,5], [5,1,1])

x = tf.random.uniform([1,4,1], seed=100)
# x = x.numpy()
y = tf.Variable(tf.random.uniform([1,2,1], minval=0.4,maxval=0.6, seed=1231))
print(y)

# print(GradientDescent(model, x, y, 100, lr=0.001))
print(BundleEntropyMethod(model, x, y, 2))