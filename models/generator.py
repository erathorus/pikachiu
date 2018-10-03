import tensorflow as tf

from layers import batch_norm, conv2d_transpose, linear


def generator1(n_samples, noise=None, dim=64):
    with tf.variable_scope('generator1', reuse=tf.AUTO_REUSE):
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        x = linear('input', 128, 8 * 4 * 4 * dim, noise)
        x = tf.reshape(x, [-1, 8 * dim, 4, 4])
        x = batch_norm('bn1', x)
        x = tf.nn.relu(x)

        x = conv2d_transpose('c2', 8 * dim, 4 * dim, 5, x)
        x = batch_norm('bn2', x)
        x = tf.nn.relu(x)

        x = conv2d_transpose('c3', 4 * dim, 2 * dim, 5, x)
        x = batch_norm('bn3', x)
        x = tf.nn.relu(x)

        x = conv2d_transpose('c4', 2 * dim, dim, 5, x)
        x = batch_norm('bn4', x)
        x = tf.nn.relu(x)

        x = conv2d_transpose('c5', dim, 3, 5, x)
        x = tf.tanh(x)

        return tf.reshape(x, [-1, 3 * dim * dim])


def generator(n_samples, noise=None, dim=64):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        x = linear('input', 128, 8 * 4 * 4 * dim, noise)
        x = tf.reshape(x, [-1, 8 * dim, 4, 4])
        x = batch_norm('bn1', x)
        x = tf.nn.relu(x)

        x = conv2d_transpose('c2', 8 * dim, 4 * dim, 5, x)
        x = batch_norm('bn2', x)
        x = tf.nn.relu(x)

        x = conv2d_transpose('c3', 4 * dim, 2 * dim, 5, x)
        x = batch_norm('bn3', x)
        x = tf.nn.relu(x)

        x = conv2d_transpose('c4', 2 * dim, dim, 5, x)
        x = batch_norm('bn4', x)
        x = tf.nn.relu(x)

        x = conv2d_transpose('c5', dim, 3, 5, x)
        x = tf.tanh(x)

        return tf.reshape(x, [-1, 3 * dim * dim])
