import tensorflow as tf 
from tensorflow import keras 
import numpy as np
import warnings


# This defines a z_i, for i > 1 layer

class layer_first_z(keras.layers.Layer):
    def __init__(self, m_1:int, activation="relu", **kwargs):
        """This is the first layer on the convex z path, hence it takes x,y arguments

        Args:
            m_1 = Output dimension of the layer
        """
        super().__init__(**kwargs) 
        self.m_1 = m_1
        self.activation = keras.activations.get(activation) 

    def build(self, input_shape):
        """ We assume that input_shape looks like:
        (n (x dimension), p (y dimension))
        """
        # Unpacking the values
        assert len(input_shape) == 2, "Input_shape dimension is {}".format(len(input_shape)) + "but expected length 2"
        n, p = input_shape
        n = n[0]
        p = p[0]
        if p != 1: #TODO: Test if this warning is really necessary
            warnings.warn("y, i.e. the action parameter shape, is not 1, it is {}".format(p))


        # Define the weights 
        self.W_0_y = self.add_weight(
            shape=[self.m_1,p],
            name="W_0_y"
        )

        self.W_0_yu = self.add_weight(
            shape=[p,n],
            name="W_0_yu"
        )

        self.b_0_y = self.add_weight(
            shape=[p,1],
            name="b_0_y"
        )

        self.W_0_u = self.add_weight(
            shape=[self.m_1,n],
            name="W_0_u"
        )

        self.b_0 = self.add_weight(
            shape=[self.m_1,1],
            name="b_0"
        )
        # This needs to be the final line, to tell the parent class
        # that the model is build, sets build=True
        super().build(input_shape)

    def call(self, input):
        """Assume the input looks like:
        (x,y)"""
        # Unpack the input
        assert len(input) == 2, "Input dimension is {}".format(len(input_shape)) + "but expected length 2"
        x,y  = input
        first_summand = tf.matmul(self.W_0_y, tf.multiply(y, tf.matmul(self.W_0_yu, x) + self.b_0_y))
        assert first_summand.shape == [self.m_1, 1], "first summand shape is {}".format(first_summand.shape) + " expected it to be {}".format([self.m_1, 1])
        second_summand = tf.matmul(self.W_0_u, x) + self.b_0
        assert second_summand.shape == [self.m_1, 1], "second summand shape is {}".format(second_summand.shape) + " expected it to be {}".format([self.m_1, 1])
        output = first_summand + second_summand 
        assert output.shape == [self.m_1, 1], "output shape is {}".format(output.shape) + " expected it to be {}".format([self.m_1, 1])
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        return [self.m_1, 1]

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "m_1": self.m_2, "activation":  keras.activations.serialize(self.activation)}


if __name__ == "__main__":

    # Instantiate the layer
    layer = layer_first_z(100)

    # Ensure that the input tensors are floats, i.e. dont forget the point .
    x = tf.Variable([[1.], [1.], [1.], [1.]])
    y = tf.Variable([[1.]])
    print(y.shape)


    X_train = (x, y)

    print("The output is", layer(X_train))

    # with tf.GradientTape() as tape:
    #     Z = layer(X_train)

    # print("The weights are:", layer.b_0)
    # print("The gradient is", tape.gradient(Z, layer.b_0))

    # # print("Shape of output", output.shape)

    # # Do a forward pass

    # # Display the variables of the model
    # # print("The variables of the layer", layer.variables)
