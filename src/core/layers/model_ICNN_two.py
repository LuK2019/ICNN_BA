import tensorflow as tf
from tensorflow import keras
import numpy as np
import warnings

from .layer_inner_z import layer_inner_z
from .layer_first_z import layer_first_z
from .layer_path_u import layer_path_u


class model_ICNN_two(keras.models.Model):
    def __init__(
        self,
        layer_params_u: list,
        layer_params_z: list,
        activation_func=keras.layers.LeakyReLU(alpha=0.01),
        weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.4),
        **kwargs
    ):
        """Idea: For each layer the provided list specifies the number of units
        of the respective layer

        Args:
            layer_params_u: list of how many units the layers in the u_path have
            layer_parms_z: list of how many units the layers in the z_path have
            
        2 two layer PICNN

        This is a float32 model, all inputs need to have this dtype, b.c. all the
        weights are float32
        """
        super().__init__(**kwargs)
        # Check validity of the model parameters
        if layer_params_z[-1] != 1:
            warnings.warn(
                "The output dimension of the model is {}, for convexitvy to make sense, it should be 1".format(
                    layer_params_z[-1]
                )
            )
        assert (len(layer_params_u) == 1) & (
            len(layer_params_z) == 2
        ), "You used an unspecified number of layers, sofar we only completed the test for two layers which means u_1, u_2, z_1, z_2, z_3"
        # Initialize the layers with parameters
        self.u_1_layer = layer_path_u(
            layer_params_u[0],
            activation=activation_func,
            weight_initializer=weight_initializer,
            name="u_1",
        )
        self.z_1_layer = layer_first_z(
            layer_params_z[0],
            activation=activation_func,
            weight_initializer=weight_initializer,
            name="z_1",
        )
        self.z_2_layer = layer_inner_z(
            layer_params_z[1],
            activation=activation_func,
            weight_initializer=weight_initializer,
            name="z_2",
        )

    def call(self, input):
        """ Convention for the input shape:
            1. Input must be a tuple (x,y)
            2. x are of type tf.tensor
            2. y can be tf.tensor or tf.Variable
            3. x,y are of dtype tf.float32
            4. x has shape: [batch_size, n, 1], i.e. a 3D list of column vectors
            5. y has shape: [batch_size, m, 1], i.e. a 3D list of column vectors

            Warning: The model won't check for input correctness, you have to use
            assert utils.check_model_input(argument, y_is_var=True/False), before calling the model with the data

            Returns:
                tf.tensor, dtype=float32, shape [batch_size, layer_params_z[-1], 1] (even if y is tf.Variable, it returns tf.tensor)
        """

        # Unpack the input
        x, y = input

        # Initialize the graph sequence
        u_1 = self.u_1_layer(x)
        z_1 = self.z_1_layer((x, y))
        z_2 = self.z_2_layer((z_1, u_1, y))

        return z_2

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "layer_params_u": self.layer_params_u,
            "layer_params_z": self.layer_params_z,
        }


if __name__ == "__main__":
    model = model_ICNN_two([1, 1], [1, 1, 1])

    X_train_batch = (tf.random.uniform([10, 3, 1]), tf.random.uniform([10, 2, 1]))

    output = model(X_train_batch)
    print("The output is", output, "of type ", type(output))

    with tf.GradientTape() as tape:
        output = model(X_train_batch)

    theta = model.trainable_variables

    gradient = tape.gradient(output, model.trainable_variables)

    print(gradient)

    # # model.build_model()
    # # from tensorflow.python.keras import backend as K

    # # graph = K.get_session().graph
    # # writer = tf.summary.FileWriter(logdir=root_logdir, graph=graph)

    # # x = tf.Variable([[3.],[2.],[1.]])
    # # y = tf.Variable([[1.]]) # Note: It does not make a difference if we have [[1.]] or [1.]

    # # print("Does it make a difference? ", model((x, y)) == model((x, tf.Variable([1.]))))

    # # with tf.GradientTape() as tape:
    # #     pred = model((x,y))

    # # print("Model summary", model.summary())
    # # print("Length of weights", len(model.trainable_variables))
    # # print("These are the trainable variables", model.trainable_variables)
    # # gradients = tape.gradient(pred, model.trainable_variables)
    # # print("this is the gradient wrt to (the trainable)", gradients)

    # x1 = tf.Variable([[3.],[2.],[1.]])
    # y1 = tf.Variable([[1.]])

    # x2 = tf.Variable([[4.],[1.],[1.]])
    # y2 = tf.Variable([[1.]])

    # x3 = np.array([[3.],[2.],[1.]], dtype=np.float32) # When one leaves out this dtype arg we get an unnecessary warning
    # y3 = np.array([[1.]], dtype=np.float32)

    # if isinstance(x3, np.ndarray):
    #     print("The condition works")

    # print(type(x3[0][0]))

    # inputs = [(x1, y1), (x2, y2)]

    # with tf.GradientTape() as tape:
    #     pred = model((x3,y3))
    # print("Forward Pass", pred)
    # # print("Model summary", model.summary())
    # # print("Length of weights", len(model.trainable_variables))
    # # print("These are the trainable variables", model.trainable_variables)
    # # gradients = tape.gradient(pred, model.trainable_variables)
    # # print("this is the gradient wrt to (the trainable)", gradients)

    # # TO DISPLAY WITH TENSORBOARD
    # # import os
    # # logdir = os.path.join(os.curdir, "my_logs_ICNN_1_2")
    # # # @tf.function
    # # def traceme(x):
    # #     return model(x)

    # # writer = tf.summary.create_file_writer(logdir)
    # # tf.summary.trace_on(graph=True, profiler=True)
    # # # Forward pass

    # x = tf.Variable([[3.],[2.],[1.]])
    # y = tf.Variable([1.])

    # # traceme((x,y))
    # # with writer.as_default():
    # #     print("we print it:", logdir)
    # #     tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)

    # # # Run in Anaconda: tensorboard --logdir C:\Users\lukas\OneDrive\Universität\Mathematik\Bachelorarbeit\dev\my_logs_ICNN_1_2

    # # x = tf.Variable([[3.],[2.],[1.]])
    # # y = tf.Variable([1.])

    # # model.compile(loss="mse", optimizer="sgd")
    # # x = tf.Variable([[3.],[2.],[1.]])
    # # y = tf.Variable([1.])
    # # y_train = tf.Variable([3.])
    # # # Show tensorboard
    # # import os
    # # root_logdir = os.path.join(os.curdir, "my_logs_ICNN_1")
    # # tensorboard_cv = keras.callbacks.TensorBoard(root_logdir)
    # # history = model.fit([x,y], y_train, epochs=2, callbacks=[tensorboard_cv])
