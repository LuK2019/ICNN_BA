import tensorflow as tf

theta1 = tf.Variable([3.])
theta2 = tf.Variable([4.])


def f(w1, w2):
    x= w1*w2*theta1*theta2
    return x 

w1 = tf.Variable([-2.])
w2 = tf.Variable([3.])

with tf.GradientTape() as tape:
    p = f(w1, w2)

gradients = tape.gradient(p, [theta1, theta2])


print(gradients)



#######

def f(arg):
    x,y = arg
    return y[0]**2 + y[1] + x

x = tf.Variable([[3.]], dtype="float64")
y = tf.Variable([[0.3],[0.7]], dtype="float64")
with tf.GradientTape() as tape:
    f_out = f((x,y))

gradient = tape.gradient(f_out, y)

####

from layers.model_ICNN import model_ICNN
from optimization.bundle_entropy import BundleEntropyMethod

model = model_ICNN([1,1], [1,1,1])
print("Result of ICNN" , BundleEntropyMethod(model, tf.Variable([[3.],[2.],[1.]], tf.Variable([[0.5]]), del_inactive_constraints=True))


matrix = np.array([[43.33294, 85.393],[-58.06778, 108.76018]])
np.linalg.inv(matrix)