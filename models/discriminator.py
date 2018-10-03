import tensorflow as tf

from layers import batch_norm, layer_norm, conv2d, linear


def discriminator1(inputs, dim=64):
    with tf.variable_scope('discriminator1', reuse=tf.AUTO_REUSE):
        x = tf.reshape(inputs, [-1, 3, dim, dim])

        x = conv2d('c1', 3, dim, 5, x, stride=2)
        x = tf.nn.leaky_relu(x)

        x = conv2d('c2', dim, 2 * dim, 5, x, stride=2)
        x = batch_norm('bn2', x)
        x = tf.nn.leaky_relu(x)

        x = conv2d('c3', 2 * dim, 4 * dim, 5, x, stride=2)
        x = batch_norm('bn3', x)
        x = tf.nn.leaky_relu(x)

        x = conv2d('c4', 4 * dim, 8 * dim, 5, x, stride=2)
        x = batch_norm('bn4', x)
        x = tf.nn.leaky_relu(x)

        x = tf.reshape(x, [-1, 8 * 4 * 4 * dim])
        x = linear('output', 8 * 4 * 4 * dim, 1, x)

        return tf.reshape(x, [-1])


def discriminator(inputs, dim=64):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        x = tf.reshape(inputs, [-1, 3, dim, dim])

        x = conv2d('c1', 3, dim, 5, x, stride=2)
        x = tf.nn.leaky_relu(x)

        x = conv2d('c2', dim, 2 * dim, 5, x, stride=2)
        x = layer_norm('bn2', x)
        x = tf.nn.leaky_relu(x)

        x = conv2d('c3', 2 * dim, 4 * dim, 5, x, stride=2)
        x = layer_norm('bn3', x)
        x = tf.nn.leaky_relu(x)

        x = conv2d('c4', 4 * dim, 8 * dim, 5, x, stride=2)
        x = layer_norm('bn4', x)
        x = tf.nn.leaky_relu(x)

        x = tf.reshape(x, [-1, 8 * 4 * 4 * dim])
        x = linear('output', 8 * 4 * 4 * dim, 1, x)

        return tf.reshape(x, [-1])
