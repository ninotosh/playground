import unittest

import numpy as np
import tensorflow as tf


class TestRNN(unittest.TestCase):
    def test_basic_rnn_cell(self):
        """see test_basic_rnn_cell_graph.png for the graph"""
        batch_size = 1
        input_shape = [batch_size, 2]
        state_shape = [batch_size, 3]
        num_units = 4  # should be equal to state_shape[1] to be recurrent

        input_value = np.random.rand(*input_shape)
        state_value = np.random.rand(*state_shape)
        np_result = TestRNN._basic_linear(input_value, state_value, num_units)

        with tf.variable_scope('test_basic_rnn_cell', initializer=tf.ones_initializer()):
            inputs = tf.placeholder(tf.float32, input_shape, 'inputs')
            prev_state = tf.placeholder(tf.float32, state_shape, 'prev_state')

            cell = tf.contrib.rnn.BasicRNNCell(num_units)
            output_op, new_state_op = cell(inputs, prev_state)

            self.assertIsInstance(output_op, tf.Tensor)

            with tf.Session() as sess:
                tf.summary.FileWriter('/tmp/test_basic_rnn_cell', sess.graph)
                sess.run(tf.global_variables_initializer())

                output, new_state = sess.run([output_op, new_state_op],
                                             feed_dict={
                                                 inputs: input_value,
                                                 prev_state: state_value
                                             })

                self.assertIsInstance(output, np.ndarray)
                self.assertEqual(output.shape, (batch_size, num_units))
                self.assertTrue(np.array_equal(output, new_state))
                np.testing.assert_array_almost_equal(np_result, output)

    @staticmethod
    def _basic_linear(input_value, state, num_units):
        assert input_value.shape[0] == state.shape[0]

        concatenated = np.concatenate((input_value, state), axis=1)

        assert concatenated.shape == (input_value.shape[0], input_value.shape[1] + state.shape[1])

        w_shape = [concatenated.shape[1], num_units]
        w = np.ones(w_shape)
        output = np.tanh(concatenated.dot(w))

        assert output.shape == (input_value.shape[0], num_units)

        return output

    def test_basic_lstm_cell(self):
        batch_size = 1
        input_shape = [batch_size, 2]
        state_shape = [batch_size, 3]
        output_shape = [batch_size, 4]
        num_units = state_shape[1]

        input_value = np.random.rand(*input_shape)
        state_value = np.random.rand(*state_shape)
        output_value = np.zeros(output_shape)

        with tf.variable_scope('test_basic_lstm_cell', initializer=tf.ones_initializer()):
            inputs = tf.placeholder(tf.float32, input_shape, 'inputs')
            prev_state = tf.placeholder(tf.float32, state_shape, 'prev_state')
            prev_output = tf.placeholder(tf.float32, output_shape, 'prev_output')

            cell = tf.contrib.rnn.BasicLSTMCell(num_units)
            output_op, new_state_op = cell(inputs, (prev_state, prev_output))

            self.assertIsInstance(output_op, tf.Tensor)

            with tf.Session() as sess:
                tf.summary.FileWriter('/tmp/test_basic_lstm_cell', sess.graph)
                sess.run(tf.global_variables_initializer())

                output, new_state = sess.run([output_op, new_state_op],
                                             feed_dict={
                                                 inputs: input_value,
                                                 prev_state: state_value,
                                                 prev_output: output_value
                                             })

                self.assertIsInstance(output, np.ndarray)
                self.assertEqual(output.shape, (batch_size, num_units))

                self.assertIsInstance(new_state, tf.contrib.rnn.LSTMStateTuple)
                self.assertIsInstance(new_state.c, np.ndarray)
                self.assertIsInstance(new_state.h, np.ndarray)
                self.assertEqual(new_state.c.shape, (batch_size, num_units))
                self.assertEqual(new_state.h.shape, (batch_size, num_units))

                self.assertTrue(np.array_equal(output, new_state.h))


if __name__ == '__main__':
    unittest.main()
