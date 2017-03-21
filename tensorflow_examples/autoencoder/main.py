import itertools

import tensorflow as tf

from autoencoder.trainingdata import TrainingData


def main(_):
    epochs = 2000
    learning_rate = 0.3
    batch_size = 2

    input_size = 4
    layer_size = 2

    x = tf.placeholder(tf.float32, [None, input_size])
    encoded = tf.contrib.layers.fully_connected(x, layer_size,
                                                activation_fn=tf.sigmoid)
    decoded = tf.contrib.layers.fully_connected(encoded, input_size,
                                                activation_fn=tf.sigmoid)

    loss = tf.nn.l2_loss(x - decoded)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    training_data = TrainingData(input_size, batch_size=batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            for inputs in training_data:
                sess.run(train, feed_dict={x: inputs})

            training_data.renew_epoch()

        print('input: encoded, decoded')
        for test_input in itertools.product([0, 1], repeat=input_size):
            print('{}: {}, {}'.format(
                test_input,
                sess.run(encoded, feed_dict={x: [list(test_input)]}),
                sess.run(decoded, feed_dict={x: [list(test_input)]})
            ))


if __name__ == '__main__':
    tf.app.run()
