import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from ..utils.utils import check_model_input, H_tf, plot_function


def GradientDescent(
    f, x, y, K, lr=0.01, eps=0.0001, show_function=False
) -> "np.ndarray (2,1)":
    """Determine argmin_y f(x,y) over the n-dimensional unit-cube, for dim(y)=n.

    Necessary requirement: f is convex in the y argument. 
    Note: This is an approximate solution using subgradient descent in K iterations.
    We use a logarithmic boundary to enforce the solution within the unit cube. 

    Note: All calculations are done on float32 vectors

    Args:
        f: the objective function, arguments passed like f((x,y))
        x: Fixed x value of the model (Column Vector of shape (n, 1))
        y: Initial starting point (Column Vector of shape (m,1))
        K: Number of iterations        

    Returns:
        np.array of the same shape of y
    """
    assert np.all(~np.isnan(y[:, 0])), "We have a log error, nan @ y={}".format(y)
    y = tf.reshape(y, [1, 2, 1])
    y = tf.Variable(y, dtype="float32")
    x = tf.convert_to_tensor(x.reshape((1, 4, 1)), dtype="float32")
    argument = (x, y)

    def regularized_f(arg):
        y = arg[1]
        return f(arg) - H_tf(y)

    if show_function:
        plot_function(regularized_f, x)

    assert check_model_input(argument, y_is_var=True)
    for i in np.arange(K):
        # print("We are at iteration {}, yielding function value {}".format(i,regularized_f((x,y))))
        with tf.GradientTape() as tape:
            out = regularized_f((x, y))
        grads = tape.gradient(out, y)
        grads_clipped = [tf.clip_by_norm(weight_grad, 1) for weight_grad in grads]
        grads_clipped = tf.reshape(grads_clipped, (1, 2, 1))

        y = y - grads_clipped * lr
        y = tf.Variable(y)

    y_return = y.numpy().reshape((2, 1))
    return y_return


if __name__ == "__main__":
    y = tf.Variable([[0.8], [0.2]])
    lr = 0.01
    ITERATIONS = 100

    for i in np.arange(ITERATIONS):
        if i % 100 == 0:
            print(
                "We are at iteration {}, the current input is {} yielding function value {}".format(
                    i, y, H_tf(y)
                )
            )
        with tf.GradientTape() as tape:
            out = H_tf(y)
        gradient = tape.gradient(out, y)
        y = y - gradient * lr
        y = tf.Variable(y)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.01)
    optimizer.minimize(H_tf, var_list=[y])

