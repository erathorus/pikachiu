import numpy as np
import tensorflow as tf

import datasets.pokemon as pokemon
import utils
from models.discriminator import discriminator
from models.generator import generator

# ---------- Settings ---------- #
ITERS = 200000
CRITIC_ITERS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
BETA1 = 0
BETA2 = 0.9
LAMBDA = 10
DEVICE = '/gpu:0'
SAMPLES_DIR = 'result/samples/2'
DATA_DIR = '../data/pokemon/pokemon_64'
LOG_DIR = 'log/train/2'
MODELS_DIR = 'result/models/2'
# ------------------------------ #


if __name__ == '__main__':
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 64, 64])

        with tf.device(DEVICE):
            real_data = tf.reshape(2 * ((tf.cast(real_data_conv, tf.float32) / 255) - 0.5), [BATCH_SIZE, 3 * 64 * 64])
            fake_data = generator(BATCH_SIZE)

            disc_real = discriminator(real_data)
            disc_fake = discriminator(fake_data)

            gen_cost = -tf.reduce_mean(disc_fake)
            disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            alpha = tf.random_uniform(
                shape=[BATCH_SIZE, 1],
                minval=0,
                maxval=1
            )
            differences = fake_data - real_data
            interpolates = real_data + (alpha * differences)
            gradients = tf.gradients(discriminator(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            disc_cost += LAMBDA * gradient_penalty

            tf.summary.scalar('gen_cost', gen_cost)
            tf.summary.scalar('disc_cost', disc_cost)
            tf.summary.scalar('w_distance', tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake))

        gen_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2) \
            .minimize(gen_cost,
                      var_list=tf.get_collection(
                          tf.GraphKeys.GLOBAL_VARIABLES,
                          scope='generator'),
                      colocate_gradients_with_ops=True)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2) \
            .minimize(disc_cost,
                      var_list=tf.get_collection(
                          tf.GraphKeys.GLOBAL_VARIABLES,
                          scope='discriminator'),
                      colocate_gradients_with_ops=True)

        # Fixed noise for generating samples
        fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
        fixed_noise_sample = generator(BATCH_SIZE, fixed_noise)


        def generate_image(iteration):
            samples = session.run(fixed_noise_sample)
            samples = ((samples + 1.) * (255.99 / 2)).astype('int32')
            utils.save_images(samples.reshape((BATCH_SIZE, 3, 64, 64)), f'{SAMPLES_DIR}/samples_{iteration}.jpg')


        dataset_gen = pokemon.load(DATA_DIR, BATCH_SIZE)


        def inf_dataset_gen():
            while True:
                for images in dataset_gen():
                    yield images


        # Save a batch of ground truth samples
        _x = next(inf_dataset_gen())
        _x_r = session.run(real_data, feed_dict={real_data_conv: _x[:BATCH_SIZE]})
        _x_r = ((_x_r + 1.) * (255.99 / 2)).astype('int32')
        utils.save_images(_x_r.reshape((BATCH_SIZE, 3, 64, 64)), f'{SAMPLES_DIR}/samples_groundtruth.jpg')

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOG_DIR, session.graph)
        saver = tf.train.Saver()

        session.run(tf.global_variables_initializer())
        gen = inf_dataset_gen()
        for it in range(ITERS):
            if it > 0:
                _ = session.run(gen_train_op)

            for i in range(CRITIC_ITERS):
                _data = next(gen)
                summary, _, _ = session.run([merged, disc_cost, disc_train_op], feed_dict={real_data_conv: _data})
                train_writer.add_summary(summary, it)

            if it % 200 == 0:
                generate_image(it)
                saver.save(session, f'{MODELS_DIR}/model_{it}.ckpt')