import tensorflow as tf


def layer_norm(name, inputs):
    with tf.variable_scope(name):
        offset = tf.get_variable('offset', [inputs.shape[1], 1, 1], initializer=tf.zeros_initializer())
        scale = tf.get_variable('scale', [inputs.shape[1], 1, 1], initializer=tf.ones_initializer())

        mean, var = tf.nn.moments(inputs, [1, 2, 3], keep_dims=True)
        return tf.nn.batch_normalization(inputs, mean, var, offset, scale, variance_epsilon=1e-5)
