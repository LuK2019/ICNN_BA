import tensorflow as tf 
from tensorflow import keras 
import numpy as np

from how_to_custom_layer import some_layer

# Create a custom model via the subclassing method

# Inherit from parent class keras.models.Model

# You will be able to use .compile(), .fit(), .evaluate(), .predict() on this object!

class some_model(keras.models.Model):
    def __init__(self, output_dim, **kwargs):
        """Purpose of the constructor:
        Instantiate the key word arguments from the parent class, and 
        instantiate the layers
        """
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(output_dim, activation="relu")
        self.hidden2 = some_layer(output_dim)

    def call(self, inputs):
        """ Purpose of call:
        Conect the layers
        """
        Z = self.hidden1(inputs)
        return self.hidden2(Z)

if __name__ == "__main__":

    # Instantiate the model
    model = some_model(5)
    model.build(input_shape=(5,5))
    print(model.summary())
