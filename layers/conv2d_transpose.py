import tensorflow as tf

from initializers import StddevUniform


def conv2d_transpose(name, input_dim, output_dim, filter_size, inputs):
    with tf.variable_scope(name):
        filters = tf.get_variable('filters', [filter_size, filter_size, output_dim, input_dim],
                                  initializer=StddevUniform(0.02))
        output_shape = tf.stack([inputs.shape[0], output_dim, 2 * inputs.shape[2], 2 * inputs.shape[3]])
        conv = tf.nn.conv2d_transpose(
            value=inputs,
            filter=filters,
            output_shape=output_shape,
            strides=[1, 1, 2, 2],
            padding='SAME',
            data_format='NCHW'
        )

        biases = tf.get_variable('biases', output_dim, initializer=tf.zeros_initializer())

        return tf.nn.bias_add(conv, biases, data_format='NCHW')
