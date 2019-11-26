import numpy as np
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def CheckModelInput(arg, y_is_var=False):
    """the second argument y = arg[1], can be a tf.Variable
    when desired, instead of a tf.Tensor. This is needed when
    a subsequent derviative of the model w.r.t. to y is needed

    x,y have to be of shape [batch_size, n,1]
    """
    # Check if arg is tuple
    if type(arg) != tuple:
        warnings.warn(
            "The argument passed to the model is not a tuple it is of type {}".format(
                type(arg)
            )
        )
        return False
    # Check if tuple is of length 2
    if len(arg) != 2:
        warnings.warn(
            "The argument passed to model is a tuple of length {}, expected length 2".format(
                len(arg)
            )
        )
        return False
    x, y = arg
    k = 0
    for input in [x, y]:
        # Check if second element in tuple is a variable if y_is_var=True
        if (k == 1) & y_is_var:
            if not (
                isinstance(input, tf.Variable)
                or (type(input) == "float32_ref")
                or (
                    type(input)
                    == "tensorflow.python.ops.resource_variable_ops.ResourceVariable"
                )
            ):
                warnings.warn(
                    "Arguments {} of tuple is of type {}, expected type tf.Variable".format(
                        k, type(input)
                    )
                )
                return False
        else:
            # Check if elements in tuple are tf.Tensors if y_is_var=False
            if not isinstance(input, tf.Tensor):
                warnings.warn(
                    "Arguments {} of tuple is of type {}, expected type tf.Tensor".format(
                        k, type(input)
                    )
                )
                return False
        # Check if dtype is float32
        if (input.dtype != "tf.float32") & (input.dtype != "float32"):
            warnings.warn(
                "Argument {} is of dtype {}, expected type tf.float32".format(
                    k, input.dtype
                )
            )
            return False
        # Check if the input shape is correct
        if len(input.shape) != 3:
            warnings.warn(
                "Argument {} of tuple is of length {}, expected length 3".format(
                    k, len(input.shape)
                )
            )
            return False
        if input.shape[2] != 1:
            warnings.warn(
                "Argument {} has shape {}, expected a list of row vectors like [batch_size, n, 1]".format(
                    k, input.shape
                )
            )
            return False
        k += 1
    if x.shape[0] != y.shape[0]:
        warnings.warn(
            "Arguments have a different batch_size x: {}, y: {}".format(
                x.shape, y.shape
            )
        )
        return False
    return True


def BarrierH_tf(y):  # TODO: Test this
    """Barrier function H, tensorflow version, i.e. differentiable
    Args:
        y: tensorflow variable of shape [1,n,1]
    Returns:
        tf.tensor
    """
    y = y[0]
    inner = tf.multiply(y, tf.math.log(y)) + tf.multiply(1.0 - y, tf.math.log(1.0 - y))
    out = -tf.reduce_sum(inner)
    return -tf.reduce_sum(
        tf.multiply(y, tf.math.log(y)) + tf.multiply(1.0 - y, tf.math.log(1.0 - y))
    )


def BarrierH(y, eps=0.0001, return_numpy=True):  # TODO: Test this
    """Barrier H function with numpy computations. 

    Args:
        y: array of shape [2,1]
        [eps=0.0001 threshold such that componentens of the input are within [eps, 1-eps] 
        to avoid log(0) errors]
        [return_numpy=True return np.float, if false, return tf.Variable]
    Returns:
        A float 
    """
    y1, y2 = y[:, 0]
    y = np.array([y1, y2])
    # To enforce avoiding log(0), TODO:test this
    k = 0
    for element in [y1, y2]:
        if element > 1 - eps:
            y[k] = 0.99
        if element < eps:
            y[k] = 0.01
        k += 1
    if return_numpy:
        return -np.sum(y * np.log(y) + (1.0 - y) * np.log(1.0 - y))
    else:
        return tf.Variable(
            -np.sum(y * np.log(y) + (1.0 - y) * np.log(1.0 - y)), dtype="float32"
        )


# TODO: Test this
def CreateTargets(random_minibatch):
    """ Creates the scalar tensor vector of shape [len(random_minibatch), 1, 1] for the loss calculation 
    from the random_minibatch dataframe"""

    H_target = np.array(
        [
            BarrierH(transition["action"])
            for index, transition in random_minibatch.iterrows()
        ]
    )

    if np.any(np.isnan(H_target)):
        v = np.isnan(H_target)
        tran = [
            transition["action"] for index, transition in random_minibatch.iterrows()
        ]
        print("These are the transitions which created a NaN value", tran[v])
        raise warnings.warn(
            "We have actions smaller than 0 / larger 1 => This is error."
        )

    y_m_target = np.array(
        [transition["y_m"] for index, transition in random_minibatch.iterrows()]
    )

    y_target = H_target - y_m_target

    y_target = y_target.reshape((len(y_target), 1, 1))

    if np.any(np.isnan(y_target)):
        warnings.warn(
            "There is a np.isnan value in y_target: {}, H_target {}, action {}".format(
                y_target, H_target, random_minibatch["action"]
            )
        )
    return tf.convert_to_tensor(y_target, dtype="float32")


# TODO: Test this
def CreateArguments(random_minibatch):
    """ Creates the argument tuple (tf.tensor [len(random_minibatch), 4, 1], tf.tensor [len(random_minibatch), 2, 1], ) for self.negQ for the loss calculation from 
    the random_minibatch dataframe
    """
    action_array = np.array(
        [row["action"] for index, row in random_minibatch.iterrows()]
    )
    state_array = np.array(
        [row["current_state"] for index, row in random_minibatch.iterrows()]
    )
    return (
        tf.convert_to_tensor(state_array, dtype="float32"),
        tf.convert_to_tensor(action_array, dtype="float32"),
    )


# TODO: Test this
def PlotFunction(
    f, x, optimum_arg=None, BEGIN=0.01, END=0.99, GRANULARITY=0.05, regularized=False
):
    X = np.arange(BEGIN, END, GRANULARITY)
    Y = np.arange(BEGIN, END, GRANULARITY)
    XM, YM = np.meshgrid(X, Y)
    output = XM.copy()
    for i in np.arange(int((END - BEGIN) / GRANULARITY)):
        for j in np.arange(int((END - BEGIN) / GRANULARITY)):
            print("i,j", (i, j))
            y = tf.Variable([[XM[i, j]], [YM[i, j]]], dtype="float32")
            y = tf.reshape(y, [1, 2, 1])
            x = tf.convert_to_tensor(x)
            x = tf.reshape(x, [1, 4, 1])
            argument = (x, y)
            assert CheckModelInput(argument)
            if regularized:
                assert (0 < BEGIN < 1) and (
                    0 < END < 1
                ), "The regularized plotting is only available for the unit cube [0,1]^2, you specified [{}, {}]".format(
                    BEGIN, END
                )

                def regularized_f(arg):
                    y = arg[1]
                    return f(arg) - BarrierH_tf(y)

                out = regularized_f(argument).numpy()
            else:
                out = f(argument).numpy()
            out = out[0, 0, 0]
            assert isinstance(out, np.float32)
            output[i, j] = out
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(XM, YM, output, cmap=cm.coolwarm)
    if optimum_arg is not None:
        x = tf.convert_to_tensor(x)
        optimum_val = f((x, optimum_arg))
        optimum_val = optimum_val[0, 0, 0].numpy()
        ax.scatter(
            optimum_arg[0, :, 0][0].numpy(),
            optimum_arg[0, :, 0][1].numpy(),
            optimum_val,
        )
    plt.show()


def grad(model: "ICNN_model", argument_model: "tuple", targets) -> "loss_value, grad":
    """Compute a the gradient of the loss of model w.r.t. to target
    Args:
        model = ICNN_model
        argument_model = argument tuple for ICNN_model
        targets = right hand side target value for the MSE loss

    Returns:
        loss_value: tf.tensor with loss at current minibatch
        grad: partial derivative of the loss w.r.t. to weights of model

        Warning: Ensure that argument_model suffices the conventions
        as validated by CheckModelInput before serving the argument
    """
    with tf.GradientTape() as tape:
        output = model(argument_model)
        loss_value = tf.reduce_mean((output - targets) ** 2)
    grad = tape.gradient(loss_value, model.trainable_variables)
    return loss_value, grad


def model_loader(model_raw, PATH_TO_WEIGHTS):
    argument = (
        tf.random.uniform(shape=[1, 4, 1], minval=0, maxval=1),
        tf.random.uniform(shape=[1, 2, 1], minval=0, maxval=1),
    )
    # assert CheckModelInput(argument)
    out = model_raw(argument)
    print("This is an old weight:", model_raw.trainable_weights[0])
    model_raw.load_weights(PATH_TO_WEIGHTS)
    print("This is the new weight:", model_raw.trainable_variables[0])
    return model_raw


if __name__ == "__main__":
    for i in range(100):
        print(
            "Current_epsilon", GreedyEstimator(100, i, 0.2, 0.1, 0.8), "at episode", i
        )

