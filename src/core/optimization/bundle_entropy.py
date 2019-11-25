import tensorflow as tf
import numpy as np

from func_timeout import func_timeout, FunctionTimedOut

from . import projected_newton
from ..utils import utils
from . import pdipm


def BundleEntropyMethod(
    f,
    x: "np.ndarray (4,1)",
    y: "np.ndarry (2,1)",
    K: "int",
    solver=pdipm.pdipm_boyd,
    show_plot=False,
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
        solver: The solver you want to use    

    Returns:
        np.ndarray of the same shape of y
    """
    duplicate_counter = 0
    # Initialize the lists
    G_l = []
    h_l = []
    x = tf.reshape(tf.convert_to_tensor(x), [1, 4, 1])

    ########### START ITERATIONS ###########

    for k in np.arange(K):
        test_arg = (x, tf.reshape(y, [1, 2, 1]))

        # Calculate the gradient w.r.t to y, ensuring that it is a tf.tensor
        if isinstance(y, np.ndarray) or isinstance(y, tf.Tensor):
            # Case: y is not already a tf.Variable
            y_variable = tf.Variable(y, dtype="float32")
            x_tensor = tf.convert_to_tensor(x, dtype="float32")
            argument = (x_tensor, y_variable)
            assert utils.check_model_input(argument, y_is_var=True)

            # Calculate the gradient
            with tf.GradientTape() as tape:
                # Evaluate the gradient of f((x, . )) w.r.t. to the second argument at y
                f_out = f(argument)
            gradient = tape.gradient(f_out, argument[1])
            # Manipulate the shape of the gradient
            gradient = gradient[0, :, :]

        else:
            # Case: y is already a tf.Variable
            # Ensure correct shape, or correct it
            if y.shape != [1, 2, 1]:
                y = tf.reshape(y, [1, 2, 1])
                y = tf.Variable(y)
            argument = (x, y)
            assert utils.check_model_input(argument, y_is_var=True)
            with tf.GradientTape() as tape:
                # Evaluate the gradient of f((x, . )) w.r.t. to the second argument at y
                f_out = f(argument)
            gradient = tape.gradient(f_out, argument[1])
            # Manipulate the shape of the gradient
            gradient = gradient[0, :, :]

        # Safe the gradient in G and h in the correct shape
        gradient = tf.transpose(gradient)
        y_in = y[0, :, :]
        f_out_in = f_out[0, :, :]
        h_input = tf.transpose(
            f_out_in - tf.reduce_sum(tf.multiply(gradient, y_in))
        )  # masking np.dot, we need to slice y to reduce the first dimension [1,m,1]->[m,1]
        G_l.append(
            gradient[0]
        )  # The 0 index is there to remove the outer bracket to reduce it from dim (1,m) -> (m,)
        h_l.append(h_input[0])

        # Note: At the moment we have list of row tensors, i.e. with shapes (n,) each
        a_k = len(G_l)

        # Convert list to tf.Tensors, i.e. each element of the list becomes a row in a matrix
        G = tf.stack(G_l)
        h = tf.stack(h_l)

        # Solve for lambda using ProjNewtonLogistic
        # To send it to solver we have to convert the shape of h from (n,1) -> (n,)
        h = tf.transpose(h)[0]

        # To send it to solver we have to convert the tensors to np.arrays
        G = G.numpy()
        h = h.numpy()

        # Check if G is linearly independent
        if k > 0:
            G_unique = np.unique(G, axis=0)
            if G_unique.shape != G.shape:
                duplicate_counter += 1
                if duplicate_counter > K * 0.5:
                    if show_plot:
                        print(
                            "The shape of the function suggests that we are at a plane"
                        )
                        print("Return current best value")
                        print("Check funciton")
                        utils.plot_function(
                            f, x, optimum_arg=y, BEGIN=0.01, END=0.99, GRANULARITY=0.02
                        )

                    y = tf.reshape(y, [2, 1])
                    y = y.numpy()
                    return y

        # For the first iteration the value of lam is 1
        if a_k == 1:
            lam = np.array([[1.0]])
        # For the following iterations we calculate lam with the solver
        else:
            try:
                try:
                    y_s, lam = func_timeout(
                        1, solver, args=(G, h)
                    )  # lam is a np.array of shape (k,)
                except FunctionTimedOut:
                    print("Solver timed out.")
                    raise TimeoutError("Solver timed out.")
            # If ProjNewtonLogistic fails, we resort to GradientDescent
            except:
                print("Solver did not work. G: {}, h: {}".format(G, h))
                break

        # Convert lam to a column vector for the calculations below

        lam = lam.reshape((lam.shape[0], 1))

        # Now solve for the new solution and substitute y
        # Cast it to tf.Variable to avoid type error

        y_np = 1 / (1 + np.exp(np.dot(np.transpose(G), lam)))  # Now y is a np.ndarray
        y = tf.Variable([y_np], dtype="float32")  # Now y is a tf.Variable

        # Delete inactive constraints
        G_l = [G_l[i] for i in np.arange(len(G_l)) if lam[i] > 0]
        h_l = [h_l[i] for i in np.arange(len(h_l)) if lam[i] > 0]

    ####### END: ITERATIONS ##########

    # Return the solution as np.ndarray (2,1)
    out = y.numpy().reshape((2, 1))
    assert isinstance(
        out, np.ndarray
    ), "BundleEntropyMethod is supposed to return np.ndarray, you returned {}".format(
        type(out)
    )
    return out
