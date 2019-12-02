import tensorflow as tf
import numpy as np
from core.layers.modelpicnnthree import ModelPICNNThree
from core.layers.modelpicnntwo import ModelPICNNTwo
from core.utils.utils import PlotFunction
import matplotlib.pyplot as plt
from core.utils.utils import CheckModelInput

### MODEL WITH 2 LAYERS ###

model2 = ModelPICNNTwo(
    [10],
    [10, 1],
    weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
)
x = tf.random.uniform([1, 4, 1], minval=-20, maxval=30)
y_tensor = tf.random.uniform([1, 2, 1])
X_train_batch_tensor = (x, y_tensor)
CheckModelInput(X_train_batch_tensor)


# PlotFunction(model2, x, BEGIN=-10, END=10, GRANULARITY=0.25, regularized=False)
# PlotFunction(model2, x, BEGIN=0.1, END=0.99, GRANULARITY=0.025, regularized=True)

list_avg = []

for k in np.arange(start=100, stop=200, step=10):
    print("Current std:", k * 0.1)
    k_avg = []
    for _ in range(10):
        model2 = ModelPICNNTwo(
            [k],
            [k, 1],
            weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
        )
        k_avg.append(model2(X_train_batch_tensor))
    list_avg.append(np.average(k_avg))
    print("Average", np.average(k_avg))

plt.plot(np.arange(start=100, stop=200, step=10), list_avg)
plt.show()

y_tensor = tf.random.uniform([1, 2, 1])
y_variable = tf.Variable(y_tensor)

X_train_batch_tensor = (x, y_tensor)
X_train_batch_variable = (x, y_variable)

CheckModelInput(X_train_batch_tensor)
CheckModelInput(X_train_batch_variable)

output_t = model2(X_train_batch_tensor)
output_v = model2(X_train_batch_variable)


with tf.GradientTape() as tape:
    output = model2(X_train_batch_variable)

theta = X_train_batch_variable[1]

gradient = tape.gradient(output, theta)

print(gradient)


### MODEL WITH 3 LAYERS ###

model3 = ModelPICNNThree([10, 10], [10, 10, 1])
# x = tf.random.uniform([1, 4, 1], minval=-20, maxval=30)
# PlotFunction(model3, x, BEGIN=-10, END=10, GRANULARITY=0.25, regularized=False)
# PlotFunction(model3, x, BEGIN=0.1, END=0.99, GRANULARITY=0.025, regularized=True)


# y_tensor = tf.random.uniform([100, 2, 1])
# y_variable = tf.Variable(y_tensor)

# X_train_batch_tensor = (x, y_tensor)
# X_train_batch_variable = (x, y_variable)
output_t = model3(X_train_batch_tensor)
output_v = model3(X_train_batch_variable)


with tf.GradientTape() as tape:
    output = model3(X_train_batch_variable)

theta = X_train_batch_variable[1]

gradient = tape.gradient(output, theta)

print(gradient)