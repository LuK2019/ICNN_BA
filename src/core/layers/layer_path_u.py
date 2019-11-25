import tensorflow as tf
from tensorflow import keras
import numpy as np
import warnings


class layer_path_u(keras.layers.Layer):
    def __init__(
        self,
        n_1: int,
        activation=keras.layers.LeakyReLU(alpha=0.01),
        weight_initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0),
        **kwargs
    ):
        """This is an inner layer fully conncected, not necessarily convex path, hence it only takes 
        x, and the previous u_i's as input
        Args:
            n_1 = Output dimension of the layer, note that the last z
            layer needs output dimension 1
        """
        super().__init__(**kwargs)
        self.n_1 = n_1
        self.activation = activation
        self.weight_initializer = weight_initializer

    def build(self, input_shape):
        """ Assumptions about input shape:
         input_shape looks like: [batch_size, n, 1], i.e. a 3D list of column vectors
         input is the x component of (x,y) tuple
        """

        # Unpacking the values
        n_shape = input_shape
        n = n_shape[1]

        # Define the weights
        self.W_1 = self.add_weight(
            initializer=self.weight_initializer, shape=[self.n_1, n], name="W_1"
        )

        self.b_1 = self.add_weight(
            initializer=self.weight_initializer, shape=[self.n_1, 1], name="b_1"
        )
        super().build(input_shape)

    def call(self, input):
        """ Assumptions about input:
        x = input
        1. x is of type tf.tensor
        2. x is of dtype tf.float32
        3. x has shape: [batch_size, n, 1], i.e. a 3D list of column vectors

            Warning: The layer won't check for input correctness, you have to use
            assert utils.check_model_input(argument), before calling the model with the data

            Returns:
                tf.tensor of shape [batch_size, n_1, 1]
        """
        x = input
        output = tf.matmul(self.W_1, x) + self.b_1
        assert output.shape == [
            input.shape[0],
            self.n_1,
            1,
        ], "output shape is {}".format(output.shape) + " expected it to be {}".format(
            [self.n_1, 1]
        )
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], self.n_1, 1])

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "n_1": self.n_1,
            "activation": keras.activations.serialize(self.activation),
        }


if __name__ == "__main__":

    # Instantiate the layer
    layer = layer_path_u(100)

    X_train_batch = tf.random.uniform([10, 3, 1])

    output = layer(X_train_batch)
    print("The output is", output, "of type ", type(output))

    with tf.GradientTape() as tape:
        output = layer(X_train_batch)

    theta = layer.trainable_variables

    gradient = tape.gradient(output, layer.trainable_variables)

    print(gradient)

    # # Ensure that the input tensors are floats, i.e. dont forget the point .
    # x = tf.Variable([[1.], [1.], [1.], [1.]])

    # X_train = (x)

    # print("The output is", layer(X_train))

    # with tf.GradientTape() as tape:
    #     Z = layer(X_train)

    # print("The weights are:", layer.b_1)
    # print("The gradient is", tape.gradient(Z, layer.b_1))
