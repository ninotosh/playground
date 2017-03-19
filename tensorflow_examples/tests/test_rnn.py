import unittest

import numpy as np
import tensorflow as tf


class TestRNNCells(unittest.TestCase):
    def test_basic_rnn_cell(self):
        """see test_basic_rnn_cell.png for the graph"""
        batch_size = 1
        input_shape = [batch_size, 2]
        state_shape = [batch_size, 3]
        num_units = 4  # should be equal to state_shape[1] to be recurrent

        input_value = np.random.rand(*input_shape)
        state_value = np.random.rand(*state_shape)
        np_result = TestRNNCells._basic_linear(input_value, state_value, num_units)

        with tf.Session() as sess:
            with tf.variable_scope('test_basic_rnn_cell', initializer=tf.ones_initializer()):
                inputs = tf.placeholder(tf.float32, input_shape, 'inputs')
                prev_state = tf.placeholder(tf.float32, state_shape, 'prev_state')

                cell = tf.contrib.rnn.BasicRNNCell(num_units)
                output_op, new_state_op = cell(inputs, prev_state)

                self.assertIsInstance(output_op, tf.Tensor)

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

        with tf.Session() as sess:
            with tf.variable_scope('test_basic_lstm_cell', initializer=tf.ones_initializer()):
                inputs = tf.placeholder(tf.float32, input_shape, 'inputs')
                prev_state = tf.placeholder(tf.float32, state_shape, 'prev_state')
                prev_output = tf.placeholder(tf.float32, output_shape, 'prev_output')

                cell = tf.contrib.rnn.BasicLSTMCell(num_units)
                output_op, new_state_op = cell(inputs, (prev_state, prev_output))

                self.assertIsInstance(output_op, tf.Tensor)

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


class TestDynamicRNN(unittest.TestCase):
    @staticmethod
    def _time_steps(batch):
        time_steps = []
        for sequence in batch:
            time_step = 0
            for x in sequence:
                if x is None or any([v is None or np.isnan(v) for v in x]):
                    break
                else:
                    time_step += 1

            time_steps.append(time_step)

        return time_steps

    def test__time_steps(self):
        batch = [
            [[1, 1], [1, 0], [0, 1], [None, None]],
            [[0, 0], [0, 1], [1, 0], [1, 1]]
        ]
        self.assertListEqual(TestDynamicRNN._time_steps(batch), [3, 4])

    @staticmethod
    def _input_shape(batch):
        batch_size = len(batch)
        time_steps = TestDynamicRNN._time_steps(batch)
        longest_time_step = max(time_steps)
        feature_size = len(batch[0][0])  # every input has the same feature size
        return batch_size, longest_time_step, feature_size

    def test_run_dynamic_rnn(self):
        batch_size = 2
        longest_time_step = 4
        feature_size = 3

        batch = np.random.randn(batch_size, longest_time_step, feature_size)
        batch[0][-2:] = None  # simulate that the 1st sequence is shorter

        num_units = 5
        with tf.Session() as sess:
            for cell_class in [tf.contrib.rnn.BasicRNNCell, tf.contrib.rnn.BasicLSTMCell]:
                cell = cell_class(num_units)
                output, final_state = TestDynamicRNN.run_dynamic_rnn(sess, batch, cell)

                self.assertIsInstance(output, np.ndarray)
                self.assertEqual(output.shape, (batch_size, longest_time_step, num_units))
                self.assertEqual(output[0][-1].tolist(), [0 for _ in range(num_units)])
                self.assertEqual(output[0][-2].tolist(), [0 for _ in range(num_units)])

                if isinstance(cell, tf.contrib.rnn.BasicRNNCell):
                    self.assertIsInstance(final_state, np.ndarray)
                    self.assertEqual(final_state.shape, (batch_size, num_units))
                else:
                    self.assertIsInstance(final_state, tf.contrib.rnn.LSTMStateTuple)
                    self.assertEqual(final_state.c.shape, (batch_size, num_units))
                    self.assertEqual(final_state.h.shape, (batch_size, num_units))

                time_steps = TestDynamicRNN._time_steps(batch)
                for i in range(len(final_state)):
                    last_time_step_index = time_steps[i] - 1
                    state_output = final_state if isinstance(cell, tf.contrib.rnn.BasicRNNCell) else final_state.h
                    self.assertTrue(np.array_equal(state_output[i], output[i][last_time_step_index]))

    @staticmethod
    def run_dynamic_rnn(sess, batch, cell):
        time_steps = TestDynamicRNN._time_steps(batch)
        input_shape = TestDynamicRNN._input_shape(batch)

        with tf.variable_scope('run_dynamic_rnn', initializer=tf.ones_initializer()):
            inputs = tf.placeholder(tf.float16, input_shape, 'inputs')
            output_op, final_state_op = tf.nn.dynamic_rnn(
                cell,
                inputs,
                sequence_length=time_steps,
                dtype=tf.float16
            )

            tf.summary.FileWriter('/tmp/run_dynamic_rnn', sess.graph)
            sess.run(tf.global_variables_initializer())

            output, final_state = sess.run([output_op, final_state_op], feed_dict={inputs: batch})

        return output, final_state


if __name__ == '__main__':
    unittest.main()
