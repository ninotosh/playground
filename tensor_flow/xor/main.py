import math

import numpy as np
import tensorflow as tf

from tensor_flow.xor.trainingdata import TrainingData

LEARNING_RATE = 0.2
MAX_EPOCH = 2000
BATCH_SIZE = 1  # invariable for XOR
INPUT_SIZE = 2
HIDDEN_SIZE = 2
OUTPUT_SIZE = 1


def sigmoid(x, input_size, size):
    initializer = tf.truncated_normal_initializer(stddev=1.0/math.sqrt(input_size))
    weights = tf.get_variable('weights', [input_size, size], initializer=initializer)
    bias = tf.get_variable('bias', [size], initializer=tf.zeros)
    return tf.nn.sigmoid(tf.add(tf.matmul(x, weights), bias), name='z')


def linear(x, input_size, size):
    weights = tf.get_variable('weights', [input_size, size], initializer=tf.ones)
    bias = tf.get_variable('bias', [size], initializer=tf.zeros)
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
        return tf.merge_summary([
            tf.scalar_summary('loss', loss),
            tf.scalar_summary([['h_w_00', 'h_w_01'], ['h_w_10', 'h_w_11']], h_w),
            tf.scalar_summary(['h_b_0', 'h_b_1'], h_b),
            tf.scalar_summary([['o_w_0'], ['o_w_1']], o_w),
            tf.scalar_summary(['o_b_0'], o_b)
        ])


def summarize_epoch(test_outputs, test_targets):
    with tf.name_scope('summarize_epoch'):
        mean_error = evaluate(test_outputs, test_targets)

        return tf.merge_summary([
            tf.scalar_summary('mean diff to target', mean_error),
            tf.scalar_summary([['output_[0,0]'], ['output_[0,1]'], ['output_[1,0]'], ['output_[1,1]']], test_outputs)
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

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    summary_writer = tf.train.SummaryWriter('logdir', graph=sess.graph)

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

    print('final output of the test input:\n{}'.format(
        sess.run(predicted, feed_dict={x: test_inputs, target: test_targets}))
    )
    print('final loss after training: {}'.format(
        sess.run(loss, feed_dict={x: test_inputs, target: test_targets}))
    )
    sess.close()


if __name__ == '__main__':
    tf.app.run()
