import tensorflow as tf 
from tensorflow import keras 
import numpy as np
import warnings


# This defines a z_i, for i > 1 layer

class layer_inner_z(keras.layers.Layer):
    def __init__(self, m_2:int, activation="relu", **kwargs):
        """This is an inner layer on the convex z path, after the first u_1
        got calculated, i.e. i > 1, z_i

        Args:
            m_2 = Output dimension of the layer, note that the last z
            layer needs output dimension 1
        """
        super().__init__(**kwargs) 
        self.m_2 = m_2
        self.activation = keras.activations.get(activation) 

    def build(self, input_shape):
        """ We assume that input_shape looks like:
        (m_1 (z_1 dimension), n_1(u_1 dimension), 1 (y_dimension))
        """
        # Unpacking the values
        assert len(input_shape) == 3, "Input_shape dimension is {}".format(len(input_shape)) + "but expected length 3"
        m_1, n_1, p = input_shape
        m_1 = m_1[0]
        n_1 = n_1[0]
        p = p[0]
        if p != 1:
            warnings.warn("y, i.e. the action parameter shape, is not 1, it is {}".format(p))


        # Define the weights and biases with the corresponding dimensions
        self.W_1_z = self.add_weight(
            shape=[self.m_2, m_1],
            name="W_1_z",
            constraint=keras.constraints.NonNeg #TODO: Add this application to training loop like on p396 Hands on ML
        )

        self.W_1_zu = self.add_weight( 
            shape=[m_1, n_1],
            name="W_1_zu"
        )

        self.b_1_z = self.add_weight( 
            shape=[m_1,1],
            name="b_1_z"
        )

        self.W_1_y = self.add_weight(
            shape=[self.m_2,p],
            name="W_1_y"
        )

        self.W_1_yu = self.add_weight(
            shape=[p,n_1],
            name="W_1_yu"
        )

        self.b_1_u = self.add_weight(
            shape=[p,1],
            name="b_1_u"
        )

        self.W_1_u = self.add_weight(
            shape=[self.m_2,n_1],
            name="W_1_u"
        )

        self.b_1 = self.add_weight(
            shape=[self.m_2,1],
            name="b_1"
        )
        # This needs to be the final line, to tell the parent class
        # that the model is build, sets build=True
        super().build(input_shape)

    def call(self, input):
        """Assume the input looks like:
        (z_1, u_1, y)"""
        # Unpack the input
        assert len(input) == 3, "Input dimension is {}".format(len(input_shape)) + "but expected length 3"
        z_1, u_1, y = input

        # Start the computation
        # Clip the negative values, as done in paper
        make_positive = tf.matmul(self.W_1_zu, u_1) + self.b_1_z
        positive_parameters = tf.where(make_positive < 0, tf.zeros_like(make_positive), make_positive) #TODO: Can one differentiate this?
        # assert np.sum(positive_parameters < 0) == 0, "detected {} negative components in W_i_zu*u_1+b_1, there should only be non-negative parameters".format(np.sum(positive_parameters < 0)) + "the faulty parameter is {}".format(positive_parameters) TODO: Error:     NotImplementedError: Cannot convert a symbolic Tensor (model_icnn/layer_inner_z/Less_1:0) to a numpy array.

        assert positive_parameters.shape == z_1.shape, "positive parameters shape is {}".format(positive_parameters.shape) + " but expected it to be shape of z_1={}".format(z_1.shape)
        first_summand = tf.matmul(self.W_1_z, tf.multiply(z_1, positive_parameters))
        assert first_summand.shape == [self.m_2, 1], "first summand shape is {}".format(first_summand.shape) + " expected it to be {}".format([self.m_2, 1])
        second_summand = tf.matmul(self.W_1_y, tf.multiply(y, tf.matmul(self.W_1_yu, u_1) + self.b_1_u))
        assert second_summand.shape == [self.m_2, 1], "second summand shape is {}".format(second_summand.shape) + " expected it to be {}".format([self.m_2, 1])
        third_summand = tf.matmul(self.W_1_u, u_1) + self.b_1
        assert third_summand.shape == [self.m_2, 1], "third summand shape is {}".format(third_summand.shape) + " expected it to be {}".format([self.m_2, 1])
        output = first_summand + second_summand + third_summand
        assert output.shape == [self.m_2, 1], "output shape is {}".format(output.shape) + " expected it to be {}".format([self.m_2, 1])
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        return [self.m_2, 1]

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "m_2": self.m_2, "activation":  keras.activations.serialize(self.activation)}


if __name__ == "__main__":

    # Instantiate the layer
    layer = layer_inner_z(1)
    print(layer.m_2)
    # Ensure that the input tensors are floats, i.e. dont forget the point .
    z_1 = tf.Variable([[-1.], [1.], [1.], [-1.]])
    u_1 = tf.Variable([[3.], [2.], [1.]])
    y = tf.Variable([[1.]])

    print("z_1:", z_1)
    print("after clipping", tf.where(z_1 < 0, tf.zeros_like(z_1), z_1))


    X_train = (z_1, u_1, y)

    with tf.GradientTape() as tape:
        Z = layer(X_train)

    print("The weights are:", layer.b_1_u)
    print("The gradient is", tape.gradient(Z, layer.b_1_u))



    # print("Shape of output", output.shape)

    # Do a forward pass

    # Display the variables of the model
    # print("The variables of the layer", layer.variables)
