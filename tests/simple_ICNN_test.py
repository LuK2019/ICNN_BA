import tensorflow as tf 
from tensorflow import keras 
import numpy as np

# Define custom layer having two seperate input layers (z_i, y_i), whereas the 
# weights assosciated with the second argument only has positive weights

class cvx_path_layer_in(keras.layers.Layer):
    def __init__(self, z_1_units, activation="relu", **kwargs):
        """this is a z layer in the convex path
        z_1_units = m_1 from paper, i.e. the number of units in this layer, note the final layer will need to have sihze of 1
        
        Reference: p.383 Hands_on_ML
        """
        super().__init__(**kwargs)
        self.z_1_units = z_1_units
        self.actviation = keras.activations.get(activation)

    def build(self, batch_input_shape):
        """Purpose: Create the weights of the layer"""
        x_path_shape  = batch_input_shape.shape[0]
        cvx_input_shape = batch_input_shape[1]

        self.kernel_1 = self.add_weight(
            name="W_0^u"
            shape=[self.z_1_units, x_path_shape]
            initializer="glorot_normal" #Check which to use
        )

        self.kernel_2 = self.add_weight(
            name="b_0^y"
            shape=[cvx_input_shape]
            initializer="glorot_normal"
        )

        self.kernel_3 = self.add_weight(
            name="W_0^y"
            shape=[self.z_1_units, cvx_input_shape]
            initializer="glorot_normal"
        )

        self.kernel_4 = self.add_weight(
            name="b_0"
            shape=[self.z_1_units]
            initializer="glorot_normal"
        )

        self.kernel_5 = self.add_weight(
            name="W_0^yu"
            shape=[cvx_input_shape, x_path_shape]
            initializer="glorot_normal"
        )

        # Sets self.built = True in parent class, must be at the end of build block
        super().build(batch_input_shape)
    
    def call(self, input):
        """Purpose: Perform the desired operation (the layers logic)

        input is list or tuple with (x_path, cvx_input)
        """
        if cvx_input_shape > 1: # TODO: For y.shape = 1, use a different logic than matmul
            self.actviation(
                tf.matmul(self.kernel_3, tf.mul(input[1], tf.matmul(self.kernel_5,input[0]) + self.kernel_2) +
                tf.matmul(self.kernel_1, input[0])+self.kernel_4
            )
    
    def compute_output_shape(self):
        return tf.TensorShape([self.z_1_units,1])
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units":self.z_1_units, "activation":keras.activations.serialize(self.activation)}




class cvx_path_layer(keras.layers.Layer):




class icnn_layer_hidden(keras.models.Model):
    def __init__(self,  input_x_shape, input_cvx_shape, units=200, activation="relu", **kwargs):
        """Here we construct the layers
        shapes are integers
        """
        super().__init__(**kwargs) #Handles standard arguments
        self.input_z = kerays.layers.Input(shape=[input_x_shape], activation=activation)
        self.input_y = keras.layers.Input(shape=[input_cvx_shape], acitvation=activation)
        self.u_1 = keras.layers.Dense(units=units,activation=activation)
        self.u_2 = keras.layers.Dense(units=units,activation=activation)
        self.z_1 = cvx_layer(...)
        self.z_2 = cvx_layer(...)
        self.z_3 = cvx_layer_out(...)

    
    def call(self, inputs):
        """Here we define the conection between the layers"""



# Define a simple ICNN model
num_input_factors = 4
input = keras.layers.Input(shape=(num_input_factors,))
hidden1 = keras.layers.Dense(200, activation="relu")(input)
