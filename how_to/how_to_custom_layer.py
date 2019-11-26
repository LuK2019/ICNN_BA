import tensorflow as tf 
from tensorflow import keras 
import numpy as np

# How to create a custom keras.layer via the subclassing method

#Inherit from the keras.layers.Layer class 

# Note you cannot use this layer in the sequential API, only use the funcitional/subclassing API for creating the model

class some_layer(keras.layers.Layer):
    def __init__(self, units, activation="relu", **kwargs):
        """Purpose of the constructor:
        Sepcify the meta-parameters like size, activation functions, etc. of the layer
        """
        # Pass the key-word arguments for the parent class as well
        super().__init__(**kwargs) 

        # Instantiate the specified meta-parameters as attribute
        self.units = units

        # Get the activation function object from the the provided string
        self.activation = keras.activations.get(activation) 

    def build(self, input_shape):
        """Purpose of build:
        Instantiate the necessary weights (variables), for the layer. Here the important thing is,
        that we can infer the dimension of the required weights from the tuple input_shape (rows, columns)
        """
        # When doing a forward pass, the input_shape argument is inferred via the argument of the call method, input as input.shape
        self.kernel = self.add_weight(
            shape=[self.units, 5],
            name="W"
        )

        self.bias = self.add_weight(
            shape=[self.units, 1],
            name="bias"
        )
        # This must be at the end
        # Call parents build method, this sets build = True in keras
        super().build(input_shape)

    # USE THIS WHEN ONLY ONE INPUT AND ONE OUTPUT
    def call(self, input):
        """ Purpose of call:
        Define the layers logic, i.e. the desired computations, on the input
        """
        return self.activation(tf.matmul(self.kernel, input) + self.bias)

    def compute_output_shape(self, input_shape):
        """ Purpose:
        Return the shape of this layer like [no_rows, no_cols]
        """
        return [self.units, 1]
    # END

    # # USE THIS WHEN MULTIPLE INPUTS AND MULTIPLE OUTPUTS
    # def call(self, input_list):
    #     """ Here the input is now in a list
    #     """
    #     X1, X2 = input_list # Unpack the individual inputs 
    #     # Return multiple outputs as list
    #     return [X1 * X2, X1 + X2]

    # def compute_output_shape(self, input_shape):
    #     """ Here the input shapes are now in a list
    #     """
    #     # Unpack the list of input_shape
    #     B1, B2 = input_shape
    #     # Return multiplie shapes as list
    #     return [B1, B2]
    # # END


    def get_config(self):
        """ Purpose:
        Used when saving the model
        """
        base_config = super().get_config() # We call the config of the parent class
        # This basicly works like:
        # 1. Store the **base_config stuff
        # 2. Map the object name to the object itself, of the hyper parameter objects
        return {**base_config, "units":self.units, "activation": keras.activations.serialize(self.activation)} # We add our meta-parameters to base_config
        
    

if __name__ == "__main__":

    # Instantiate the layer
    layer = some_layer(5)

    X_train = np.array(np.zeros((5,1)))

    # Do a forward pass
    print(layer(X_train))

    # Display the variables of the model
    print(layer.variables)
