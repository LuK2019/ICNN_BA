import tensorflow as tf
from tensorflow import keras
import numpy as np
import warnings


class layer_inner_z(keras.layers.Layer):
    def __init__(
        self,
        m_2: int,
        activation=keras.layers.LeakyReLU(alpha=0.01),
        weight_initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0),
        **kwargs
    ):
        """This is an inner layer on the convex z path, after the first u_1
        got calculated, i.e. i > 1, z_i

        Args:
            m_2 = Output dimension of the layer, note that the last z
            layer needs output dimension 1
        """
        super().__init__(**kwargs)
        self.m_2 = m_2
        self.activation = activation
        self.weight_initializer = weight_initializer

    def build(self, input_shape):
        """ We assume that input_shape looks like:
            ([batch_size, m_1, 1], [batch_size, n_1, 1], [batch_size, m, 1]) = (z_1 dimension, u_1 dimension, y_dimension)
        """
        # Unpacking the values
        assert len(input_shape) == 3, (
            "Input_shape dimension is {}".format(len(input_shape))
            + "but expected length 3"
        )

        z_1_shape, u_1_shape, y_shape = input_shape
        m_1 = z_1_shape[1]
        n_1 = u_1_shape[1]
        m = y_shape[1]

        # Define the weights and biases with the corresponding dimensions
        self.W_1_z = self.add_weight(
            initializer=self.weight_initializer,
            shape=[self.m_2, m_1],
            name="W_1_z",
            constraint=keras.constraints.NonNeg(),
        )

        self.W_1_zu = self.add_weight(
            initializer=self.weight_initializer, shape=[m_1, n_1], name="W_1_zu"
        )

        self.b_1_z = self.add_weight(
            initializer=self.weight_initializer, shape=[m_1, 1], name="b_1_z"
        )

        self.W_1_y = self.add_weight(
            initializer=self.weight_initializer, shape=[self.m_2, m], name="W_1_y"
        )

        self.W_1_yu = self.add_weight(
            initializer=self.weight_initializer, shape=[m, n_1], name="W_1_yu"
        )

        self.b_1_u = self.add_weight(
            initializer=self.weight_initializer, shape=[m, 1], name="b_1_u"
        )

        self.W_1_u = self.add_weight(
            initializer=self.weight_initializer, shape=[self.m_2, n_1], name="W_1_u"
        )

        self.b_1 = self.add_weight(
            initializer=self.weight_initializer, shape=[self.m_2, 1], name="b_1"
        )
        # This needs to be the final line, to tell the parent class
        # that the model is build, sets build=True
        # Apply Constraint:
        for variable in self.variables:
            if variable.constraint is not None:
                variable.assign(variable.constraint(variable))
        super().build(input_shape)

    def call(self, input):
        """Convention for the input:
            1. Input is a triple (z_1, u_1, y)
            2. z_1, u_1, y are tf.tensors
            3. z_1, u_1, y are of dtype.float32
            4. z_1 has shape: [batch_size, m_1, 1]
            5. u_1 has shape: [batch_size, n_1, 1]
            6. y has shape: [batch_size, m, 1]

            Returns:
                tf.tensor of shape [bathc_size, m_2 (output_size), 1]
        """
        # Unpack the input
        assert len(input) == 3, (
            "Input dimension is {}".format(len(input_shape)) + "but expected length 3"
        )

        z_1, u_1, y = input

        # Start the computation
        # Clip the negative values, as done in paper
        make_positive = tf.matmul(self.W_1_zu, u_1) + self.b_1_z
        positive_parameters = tf.where(
            make_positive < 0, tf.zeros_like(make_positive), make_positive
        )  # TODO: Can one differentiate this? Test a negative case!
        # assert np.sum(positive_parameters < 0) == 0, "detected {} negative components in W_i_zu*u_1+b_1, there should only be non-negative parameters".format(np.sum(positive_parameters < 0)) + "the faulty parameter is {}".format(positive_parameters) TODO: Error:     NotImplementedError: Cannot convert a symbolic Tensor (model_icnn/layer_inner_z/Less_1:0) to a numpy array.

        assert (
            positive_parameters.shape == z_1.shape
        ), "positive parameters shape is {}".format(
            positive_parameters.shape
        ) + " but expected it to be shape of z_1={}".format(
            z_1.shape
        )
        first_summand = tf.matmul(self.W_1_z, tf.multiply(z_1, positive_parameters))
        assert first_summand.shape == [
            z_1.shape[0],
            self.m_2,
            1,
        ], "first summand shape is {}".format(
            first_summand.shape
        ) + " expected it to be {}".format(
            [self.m_2, 1]
        )
        second_summand = tf.matmul(
            self.W_1_y, tf.multiply(y, tf.matmul(self.W_1_yu, u_1) + self.b_1_u)
        )
        assert second_summand.shape == [
            z_1.shape[0],
            self.m_2,
            1,
        ], "second summand shape is {}".format(
            second_summand.shape
        ) + " expected it to be {}".format(
            [self.m_2, 1]
        )
        third_summand = tf.matmul(self.W_1_u, u_1) + self.b_1
        assert third_summand.shape == [
            z_1.shape[0],
            self.m_2,
            1,
        ], "third summand shape is {}".format(
            third_summand.shape
        ) + " expected it to be {}".format(
            [self.m_2, 1]
        )
        output = first_summand + second_summand + third_summand
        assert output.shape == [z_1.shape[0], self.m_2, 1], "output shape is {}".format(
            output.shape
        ) + " expected it to be {}".format([self.m_2, 1])
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0][0], self.m_2, 1])

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "m_2": self.m_2,
            "activation": keras.activations.serialize(self.activation),
        }


if __name__ == "__main__":

    # Instantiate the layer
    layer = layer_inner_z(10)

    X_train_batch = (
        tf.random.uniform([10, 3, 1]),
        tf.random.uniform([10, 2, 1]),
        tf.random.uniform([10, 3, 1]),
    )

    output = layer(X_train_batch)
    print("The output is", output, "of type ", type(output))

    with tf.GradientTape() as tape:
        output = layer(X_train_batch)

    theta = layer.trainable_variables

    gradient = tape.gradient(output, layer.trainable_variables)

    print(gradient)

    print(layer.m_2)
    # Ensure that the input tensors are floats, i.e. dont forget the point .
    z_1 = tf.Variable([[-1.0], [1.0], [1.0], [-1.0]])
    u_1 = tf.Variable([[3.0], [2.0], [1.0]])
    y = tf.Variable([[1.0]])

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
