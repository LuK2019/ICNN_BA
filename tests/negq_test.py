import os

import numpy as np
import tensorflow as tf
import tflearn

# import bundle_entropy
# from replay_memory import ReplayMemory
# from helper import variable_summaries



def negQ(x, y, reuse=False, szs=[10,10]):
    """Modelling the negative Q value aus PICNN.
    x = state variabel
    y = action variable, convex 
    reuse ? 
    """

    assert(len(szs) >= 1)
    fc = tflearn.fully_connected
    # bn = tflearn.batch_normalization # We don't want batch normalization
    lrelu = tflearn.activations.leaky_relu

    if reuse: # ? 
        tf.get_variable_scope().reuse_variables()

    nLayers = len(szs)
    us = []
    zs = []
    z_zs = []
    z_ys = []
    z_us = []

    reg = 'L2'

    prevU = x
    for i in range(nLayers):
        with tf.variable_scope('u'+str(i)) as s:
            u = fc(prevU, szs[i], reuse=reuse, scope=s, regularizer=reg)
            if i < nLayers-1:
                u = tf.nn.relu(u)
                # if FLAGS.icnn_bn:
                #     u = bn(u, reuse=reuse, scope=s, name='bn')
        variable_summaries(u, suffix='u{}'.format(i))
        us.append(u)
        prevU = u

    prevU, prevZ = x, y
    for i in range(nLayers+1):
        sz = szs[i] if i < nLayers else 1
        z_add = []
        if i > 0:
            with tf.variable_scope('z{}_zu_u'.format(i)) as s:
                zu_u = fc(prevU, szs[i-1], reuse=reuse, scope=s,
                            activation='relu', bias=True,
                            regularizer=reg, bias_init=tf.constant_initializer(1.))
                variable_summaries(zu_u, suffix='zu_u{}'.format(i))
            with tf.variable_scope('z{}_zu_proj'.format(i)) as s:
                z_zu = fc(tf.mul(prevZ, zu_u), sz, reuse=reuse, scope=s,
                            bias=False, regularizer=reg)
                variable_summaries(z_zu, suffix='z_zu{}'.format(i))
            z_zs.append(z_zu)
            z_add.append(z_zu)

        with tf.variable_scope('z{}_yu_u'.format(i)) as s:
            yu_u = fc(prevU, self.dimA, reuse=reuse, scope=s, bias=True,
                        regularizer=reg, bias_init=tf.constant_initializer(1.))
            variable_summaries(yu_u, suffix='yu_u{}'.format(i))
        with tf.variable_scope('z{}_yu'.format(i)) as s:
            z_yu = fc(tf.mul(y, yu_u), sz, reuse=reuse, scope=s, bias=False,
                        regularizer=reg)
            z_ys.append(z_yu)
            variable_summaries(z_yu, suffix='z_yu{}'.format(i))
        z_add.append(z_yu)

        with tf.variable_scope('z{}_u'.format(i)) as s:
            z_u = fc(prevU, sz, reuse=reuse, scope=s,
                        bias=True, regularizer=reg,
                        bias_init=tf.constant_initializer(0.))
            variable_summaries(z_u, suffix='z_u{}'.format(i))
        z_us.append(z_u)
        z_add.append(z_u)

        z = tf.add_n(z_add)
        variable_summaries(z, suffix='z{}_preact'.format(i))
        if i < nLayers:
            # z = tf.nn.relu(z)
            z = lrelu(z, alpha=FLAGS.lrelu)
            variable_summaries(z, suffix='z{}_act'.format(i))

        zs.append(z)
        prevU = us[i] if i < nLayers else None
        prevZ = z

    z = tf.reshape(z, [-1], name='energies')
    return z

if __name__ == "__main__":
    negQ(1,2)
