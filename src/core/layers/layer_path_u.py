import tensorflow as tf 
from tensorflow import keras 
import numpy as np
import warnings


# This defines a z_i, for i > 1 layer

class layer_path_u(keras.layers.Layer):
    def __init__(self, n_1:int, activation="relu", **kwargs):
        """This is an inner layer fully conncected, not necessarily convex path, hence it only takes 
        x, and the previous u_i's as input
        Args:
            n_1 = Output dimension of the layer, note that the last z
            layer needs output dimension 1
        """
        super().__init__(**kwargs) 
        self.n_1 = n_1
        self.activation = keras.activations.get(activation) 

    def build(self, input_shape):
        """ We assume that input_shape looks like:
        (n (x dimension))
        """
        print("Input shape", input_shape)
        # Unpacking the values
        assert len(input_shape) == 2, "Input_shape dimension is {} but expected length 1".format(len(input_shape))
        n = input_shape
        n = n[0]

        # Define the weights 
        self.W_1 = self.add_weight(
            shape=[self.n_1,n],
            name="W_1"
        )

        self.b_1 = self.add_weight(
            shape=[self.n_1,1],
            name="b_1"
        )

        # self. = self.add_weight(
        #     shape=[,],
        #     name=""
        # )
        super().build(input_shape)

    def call(self, input):
        """Assume the input looks like:
        (x)"""
        # Unpack the input
        assert input.shape[1] == 1, "Input dimension is {} but expected length 1".format(input.shape[1])
        x = input
        output = tf.matmul(self.W_1, x) + self.b_1
        assert output.shape == [self.n_1, 1], "output shape is {}".format(output.shape) + " expected it to be {}".format([self.n_1, 1])
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        return [self.n_1, 1]

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "n_1": self.m_2, "activation":  keras.activations.serialize(self.activation)}


if __name__ == "__main__":

    # Instantiate the layer
    layer = layer_path_u(100)

    # Ensure that the input tensors are floats, i.e. dont forget the point .
    x = tf.Variable([[1.], [1.], [1.], [1.]])

    X_train = (x)

    print("The output is", layer(X_train))

    with tf.GradientTape() as tape:
        Z = layer(X_train)

    print("The weights are:", layer.b_1)
    print("The gradient is", tape.gradient(Z, layer.b_1))
