import tensorflow as tf


def batch_norm(name, inputs):
    with tf.variable_scope(name):
        offset = tf.get_variable('offset', inputs.shape[1], initializer=tf.zeros_initializer())
        scale = tf.get_variable('scale', inputs.shape[1], initializer=tf.ones_initializer())

        outputs, _, _ = tf.nn.fused_batch_norm(inputs, scale, offset, epsilon=1e-5, data_format='NCHW')

        return outputs
