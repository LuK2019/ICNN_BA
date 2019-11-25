import tensorflow as tf
from tensorflow import keras
import numpy as np
import warnings


class layer_first_z(keras.layers.Layer):
    def __init__(
        self,
        m_1: int,
        activation=keras.layers.LeakyReLU(alpha=0.01),
        weight_initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0),
        **kwargs
    ):
        """This is the first layer on the convex z path, hence it takes x,y arguments
        
        Args: (for constructor)
            m_1 = Output dimension of the layer
            [activation= activation function object from keras.layers; default is leaky ReLU]
            [weight_initializer= weight initializer from tf. ... initializer]
        """
        super().__init__(**kwargs)
        self.m_1 = m_1
        self.activation = activation
        self.weight_initializer = weight_initializer

    def build(self, input_shape):
        """ We assume that input_shape looks like:
        ([batch_size, n, 1] (x dimension), [batch_size, m, 1] (y dimension))

        build is executed during the first inference of the model, b.c. then the input shape 
        is explicit
        """

        # Unpacking the values
        x_shape, y_shape = input_shape
        n = x_shape[1]
        m = y_shape[1]
        # Define the weights
        self.W_0_y = self.add_weight(
            initializer=self.weight_initializer, shape=[self.m_1, m], name="W_0_y"
        )

        self.W_0_yu = self.add_weight(
            initializer=self.weight_initializer, shape=[m, n], name="W_0_yu"
        )

        self.b_0_y = self.add_weight(
            initializer=self.weight_initializer, shape=[m, 1], name="b_0_y"
        )

        self.W_0_u = self.add_weight(
            initializer=self.weight_initializer, shape=[self.m_1, n], name="W_0_u"
        )

        self.b_0 = self.add_weight(
            initializer=self.weight_initializer, shape=[self.m_1, 1], name="b_0"
        )
        # This needs to be the final line, to tell the parent class
        # that the model is build, sets build=True
        super().build(input_shape)

    def call(self, input: tuple) -> "tf.Tensor":
        """ Convention for the input :
            1. Input must be a tuple (x,y)
            2. x,y are of type tf.tensor
            3. x,y are of dtype tf.float32
            4. x has shape: [batch_size, n, 1], i.e. a 3D list of column vectors
            5. y has shape: [batch_size, m, 1], i.e. a 3D list of column vectors

            Warning: The layer won't check for input correctness, you have to use
            assert utils.check_model_input(argument), before calling the model with the data
            Args:
                input = (x,y) input tuple following the above conventions
            Returns:
                tf.tensor of shape [batch_size, m_1(output_shape), 1]
            
        """
        # Unpack the input
        x, y = input
        first_summand = tf.matmul(
            self.W_0_y, tf.multiply(y, tf.matmul(self.W_0_yu, x) + self.b_0_y)
        )
        assert first_summand.shape == [
            x.shape[0],
            self.m_1,
            1,
        ], "first summand shape is {}".format(
            first_summand.shape
        ) + " expected it to be {}".format(
            [x.shape[0], self.m_1, 1]
        )
        second_summand = tf.matmul(self.W_0_u, x) + self.b_0
        assert second_summand.shape == [
            x.shape[0],
            self.m_1,
            1,
        ], "second summand shape is {}".format(
            second_summand.shape
        ) + " expected it to be {}".format(
            [self.m_1, 1]
        )
        output = first_summand + second_summand
        assert output.shape == [x.shape[0], self.m_1, 1], "output shape is {}".format(
            output.shape
        ) + " expected it to be {}".format([self.m_1, 1])
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0][0], self.m_1, 1])

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "m_1": self.m_1,
            "activation": keras.activations.serialize(self.activation),
        }


if __name__ == "__main__":

    # Instantiate the layer
    layer = layer_first_z(100)

    # Ensure that the input tensors are floats, i.e. dont forget the point .
    x = tf.Variable([[1.0], [1.0], [1.0], [1.0]])
    y = tf.Variable([[1.0]])
    print(y.shape)

    X_train = (x, y)
    X_train_batch = (tf.random.uniform([10, 3, 1]), tf.random.uniform([10, 2, 1]))

    output = layer(X_train_batch)
    print("The output is", output, "of type ", type(output))

    with tf.GradientTape() as tape:
        output = layer(X_train_batch)

    theta = layer.trainable_variables

    gradient = tape.gradient(output, layer.trainable_variables)

    print(gradient)

