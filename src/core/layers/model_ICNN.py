import tensorflow as tf 
from tensorflow import keras 
import numpy as np
import warnings

from .layer_inner_z import layer_inner_z
from .layer_first_z import layer_first_z
from .layer_path_u import layer_path_u 

class model_ICNN(keras.models.Model):
    def __init__(self, layer_params_u:list, layer_params_z:list, **kwargs):
        """Idea: For each layer the provided list specifies the number of units
        of the respective layer

        This is a float32 model, all inputs need to have this dtype, b.c. all the
        weights are float32
        """
        super().__init__(**kwargs)
        # assert (len(layer_params_u)==2) && (len(layer_params_z)==3), "You used an unspecified number of layers, sofar we only completed the test for two layers which means u_1, u_2, z_1, z_2, z_3"
        self.u_1_layer = layer_path_u(layer_params_u[0], name="u_1")
        self.u_2_layer = layer_path_u(layer_params_u[1], name="u_2")
        self.z_1_layer = layer_first_z(layer_params_z[0], name="z_1")
        self.z_2_layer = layer_inner_z(layer_params_z[1], name="z_2")
        if layer_params_z[-1]!=1:
             warnings.warn("The output dimension of the model is {}, for convexitvy to make sense, it should be 1".format(layer_params_z[-1]))
        self.z_3_layer = layer_inner_z(layer_params_z[2],name="z_3")

    def call(self, input):
        """ Assume that input has shape like (x,y)"""
        # Unpack the input
        x,y = input
        # Check if the data type is applicable
        if isinstance(x, np.ndarray):
            print("converted input x: {} from ndarray to tensor".format(x))
            x = tf.convert_to_tensor(x, dtype="float32")
        if isinstance(y, np.ndarray):
            print("converted input y: {} from ndarray to tensor".format(y))
            y = tf.convert_to_tensor(y, dtype="float32")
        u_1 = self.u_1_layer(x)
        u_2 = self.u_2_layer(u_1)
        z_1 = self.z_1_layer((x,y))
        z_2 = self.z_2_layer((z_1, u_1, y))
        z_3 = self.z_3_layer((z_2, u_2, y))
        
        return z_3 

if __name__ == "__main__":
    model = model_ICNN([1,1], [1,1,1])
    # model.build_model()
    # from tensorflow.python.keras import backend as K

    # graph = K.get_session().graph
    # writer = tf.summary.FileWriter(logdir=root_logdir, graph=graph)

    # x = tf.Variable([[3.],[2.],[1.]])
    # y = tf.Variable([[1.]]) # Note: It does not make a difference if we have [[1.]] or [1.]
 
    # print("Does it make a difference? ", model((x, y)) == model((x, tf.Variable([1.]))))

    # with tf.GradientTape() as tape:
    #     pred = model((x,y))
    
    # print("Model summary", model.summary())
    # print("Length of weights", len(model.trainable_variables))
    # print("These are the trainable variables", model.trainable_variables)
    # gradients = tape.gradient(pred, model.trainable_variables)
    # print("this is the gradient wrt to (the trainable)", gradients)

    x1 = tf.Variable([[3.],[2.],[1.]])
    y1 = tf.Variable([[1.]]) 
   
    x2 = tf.Variable([[4.],[1.],[1.]])
    y2 = tf.Variable([[1.]]) 

    x3 = np.array([[3.],[2.],[1.]], dtype=np.float32) # When one leaves out this dtype arg we get an unnecessary warning
    y3 = np.array([[1.]], dtype=np.float32)

    if isinstance(x3, np.ndarray):
        print("The condition works")

    print(type(x3[0][0]))
 
    inputs = [(x1, y1), (x2, y2)]

    with tf.GradientTape() as tape:
        pred = model((x3,y3))
    print("Forward Pass", pred)
    # print("Model summary", model.summary())
    # print("Length of weights", len(model.trainable_variables))
    # print("These are the trainable variables", model.trainable_variables)
    # gradients = tape.gradient(pred, model.trainable_variables)
    # print("this is the gradient wrt to (the trainable)", gradients)

    



    # TO DISPLAY WITH TENSORBOARD
    # import os
    # logdir = os.path.join(os.curdir, "my_logs_ICNN_1_2")
    # # @tf.function
    # def traceme(x):
    #     return model(x)


    # writer = tf.summary.create_file_writer(logdir)
    # tf.summary.trace_on(graph=True, profiler=True)
    # # Forward pass

    x = tf.Variable([[3.],[2.],[1.]])
    y = tf.Variable([1.])


    # traceme((x,y))
    # with writer.as_default():
    #     print("we print it:", logdir)
    #     tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)

    # # Run in Anaconda: tensorboard --logdir C:\Users\lukas\OneDrive\Universität\Mathematik\Bachelorarbeit\dev\my_logs_ICNN_1_2

    # x = tf.Variable([[3.],[2.],[1.]])
    # y = tf.Variable([1.])
    

    # model.compile(loss="mse", optimizer="sgd")
    # x = tf.Variable([[3.],[2.],[1.]])
    # y = tf.Variable([1.])
    # y_train = tf.Variable([3.])
    # # Show tensorboard
    # import os
    # root_logdir = os.path.join(os.curdir, "my_logs_ICNN_1")
    # tensorboard_cv = keras.callbacks.TensorBoard(root_logdir)
    # history = model.fit([x,y], y_train, epochs=2, callbacks=[tensorboard_cv])
    
