import tensorflow as tf

from initializers import StddevUniform


def linear(name, input_dim, output_dim, inputs):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [input_dim, output_dim],
                                 initializer=StddevUniform(0.02))

        if inputs.shape.ndims == 2:
            result = tf.matmul(inputs, weight)
        else:
            reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
            result = tf.matmul(reshaped_inputs, weight)
            result = tf.reshape(result, tf.stack(tf.unstack(tf.shape(inputs))[:-1] + [output_dim]))

        biases = tf.get_variable('biases', output_dim, initializer=tf.zeros_initializer())

        return tf.nn.bias_add(result, biases)
