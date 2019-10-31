import tensorflow as tf

# 1. Define the function you want to differentiate using tensorflow operations

def f(w1, w2):
    return w1*w2*2 + tf.reduce_sum([w1, w2]) 

# 2. Use tf.GradientTape() to record the operations and to compute a gradient
# - Gradient tape only works with tf.Variables (no constants or anything else)

w1 = tf.Variable(5.)
w2 = tf.Variable(3.)

with tf.GradientTape() as tape:
    z = f(w1, w2)

# 3. Now call the gradient method to compute the gradient of z w.r.t to [w1, w2], evaluated at the values of the variables

gradients = tape.gradient(z, [w1, w2])

# Note: You can only call .gradient(...) once, to use it multiple times use:

with tf.GradientTape(persistent=True) as tape:
    z = f(w1, w2)

tape.gradient(z,w1)
tape.gradient(z, [w1, w2])

# Delete it with 
del tape

# How to calculate the gradient w.r.t. all the weights? Won't execute

model 

with tf.GradientTape() as tape:
    pred = model((x,y))

gradients = tape.gradient(pred, model.trainable_variables)



