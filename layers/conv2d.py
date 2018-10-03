import tensorflow as tf

from initializers import StddevUniform


def conv2d(name, input_dim, output_dim, filter_size, inputs, stride=1):
    with tf.variable_scope(name):
        filters = tf.get_variable('filters', [filter_size, filter_size, input_dim, output_dim],
                                  initializer=StddevUniform(0.02))

        conv = tf.nn.conv2d(
            input=inputs,
            filter=filters,
            strides=[1, 1, stride, stride],
            padding='SAME',
            data_format='NCHW'
        )

        biases = tf.get_variable('biases', output_dim, initializer=tf.zeros_initializer())

        return tf.nn.bias_add(conv, biases, data_format='NCHW')
