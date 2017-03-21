import math

import numpy as np
import tensorflow as tf

from xor.trainingdata import TrainingData

LEARNING_RATE = 0.2
MAX_EPOCH = 2000
BATCH_SIZE = 2
INPUT_SIZE = 2
HIDDEN_SIZE = 2
OUTPUT_SIZE = 1


def sigmoid(x, input_size, size):
    initializer = tf.truncated_normal_initializer(stddev=1.0/math.sqrt(input_size))
    weights = tf.get_variable('weights', [input_size, size], initializer=initializer)
    bias = tf.get_variable('bias', [size], initializer=tf.zeros_initializer)
    return tf.nn.sigmoid(tf.add(tf.matmul(x, weights), bias), name='z')


def linear(x, input_size, size):
    weights = tf.get_variable('weights', [input_size, size], initializer=tf.ones_initializer)
    bias = tf.get_variable('bias', [size], initializer=tf.zeros_initializer)
    return tf.add(tf.matmul(x, weights), bias, name='z')


def infer(x):
    with tf.variable_scope('hidden_layer'):
        hidden_output = sigmoid(x, INPUT_SIZE, HIDDEN_SIZE)

    with tf.variable_scope('output_layer'):
        final_output = linear(hidden_output, HIDDEN_SIZE, OUTPUT_SIZE)

        return final_output


def calculate_loss(outputs, targets):
    with tf.name_scope('calculate_loss'):
        # loss = tf.abs(targets - outputs)  # this is wrong
        loss = tf.nn.l2_loss(targets - outputs, name='loss')
        return loss


def train(loss, learning_rate, global_step=None):
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op


def evaluate(predicted, targets):
    with tf.name_scope('evaluate'):
        errors = tf.abs(targets - predicted, name='errors')
        mean_error = tf.reduce_mean(errors, name='mean_error')

        return mean_error


def summarize_step(loss):
    with tf.variable_scope('infer', reuse=True):
        with tf.variable_scope('hidden_layer', reuse=True):
            h_w = tf.get_variable('weights', [INPUT_SIZE, HIDDEN_SIZE])
            h_b = tf.get_variable('bias', [HIDDEN_SIZE])

        with tf.variable_scope('output_layer', reuse=True):
            o_w = tf.get_variable('weights', [HIDDEN_SIZE, OUTPUT_SIZE])
            o_b = tf.get_variable('bias', [OUTPUT_SIZE])

    with tf.name_scope('summarize_step'):
        return tf.summary.merge([
            tf.summary.scalar('loss', loss),
            tf.summary.scalar('h-w-0-0', h_w[0][0]),
            tf.summary.scalar('h-w-0-1', h_w[0][1]),
            tf.summary.scalar('h-w-1-0', h_w[1][0]),
            tf.summary.scalar('h-w-1-1', h_w[1][1]),
            tf.summary.scalar('h-b-0', h_b[0]),
            tf.summary.scalar('h-b-1', h_b[1]),
            tf.summary.scalar('o-w-0-0', o_w[0][0]),
            tf.summary.scalar('o-w-1-0', o_w[1][0]),
            tf.summary.scalar('o_b', o_b[0])
        ])


def summarize_epoch(test_outputs, test_targets):
    with tf.name_scope('summarize_epoch'):
        mean_error = evaluate(test_outputs, test_targets)

        return tf.summary.merge([
            tf.summary.scalar('mean diff to target', mean_error),
            tf.summary.scalar('output-0-0', test_outputs[0][0]),
            tf.summary.scalar('output-1-0', test_outputs[1][0])
        ])


def main(_):
    training_data = TrainingData([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]])
    test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype(float)
    test_targets = np.array([[0], [1], [1], [0]]).astype(float)

    x = tf.placeholder(tf.float32, [None, INPUT_SIZE], name='x')
    target = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name='target')
    test_x = tf.placeholder(tf.float32, [None, INPUT_SIZE], name='test_x')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    with tf.variable_scope('infer'):
        predicted = infer(x)

    loss = calculate_loss(predicted, target)
    train_op = train(loss, LEARNING_RATE, global_step)

    with tf.name_scope('summarize'):
        merge_step = summarize_step(loss)

        with tf.variable_scope('infer', reuse=True):
            test_outputs = infer(test_x)

        merge_epoch = summarize_epoch(test_outputs, test_targets)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    summary_writer = tf.summary.FileWriter('logdir', graph=sess.graph)

    for epoch in range(MAX_EPOCH):
        while True:
            inputs, targets = training_data.next_batch(BATCH_SIZE)
            if len(inputs) == 0:
                break

            feed_dict = {x: inputs, target: targets}
            sess.run(train_op, feed_dict=feed_dict)

            summary_writer.add_summary(
                sess.run(merge_step, feed_dict=feed_dict),
                tf.train.global_step(sess, global_step)
            )

        summary_writer.add_summary(
            sess.run(merge_epoch, feed_dict={test_x: test_inputs}),
            tf.train.global_step(sess, global_step)
        )

        training_data.renew_epoch()

    print('final output of the test input:')
    for (i, o) in zip(test_inputs, sess.run(predicted, feed_dict={x: test_inputs, target: test_targets})):
        print('{} => {}'.format(i, o))

    print('final loss after training: {}'.format(
        sess.run(loss, feed_dict={x: test_inputs, target: test_targets}))
    )
    sess.close()


if __name__ == '__main__':
    tf.app.run()
