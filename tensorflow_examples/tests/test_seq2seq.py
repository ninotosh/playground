import unittest

import numpy as np
import tensorflow as tf


def assert_version(target_version):
    target_major, target_minor, target_patch = map(int, target_version.split('.'))
    tf_version = tf.__version__
    if '-' in tf_version:
        tf_version = tf_version[:tf_version.find('-')]
    major, minor, patch = map(int, tf_version.split('.'))
    assert major >= target_major
    if major == target_major:
        assert minor >= target_minor
        if minor == target_minor:
            assert patch >= target_patch


class TestSeq2Seq(unittest.TestCase):
    def test_inference(self):
        assert_version('1.0.1')

        batch_size = 1
        input_depth = 1

        vocabulary_size = 3
        cell_depth = 3
        assert cell_depth == vocabulary_size

        embedding = np.array([[.1], [.2], [.3]], dtype=np.float32)
        assert embedding.shape == (vocabulary_size, input_depth)

        start_tokens = [0]
        assert len(start_tokens) == batch_size
        for token in start_tokens:
            assert 0 <= token < vocabulary_size

        end_token = 1
        assert isinstance(end_token, int)
        assert 0 <= end_token < vocabulary_size

        maximum_iterations = 10  # maximum allowed number of decoding steps
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=embedding,
            start_tokens=start_tokens,
            end_token=end_token
        )

        with tf.Session() as sess:
            encoder_final_state = tf.constant([[.3, .2, .1]])  # a.k.a. thought vector
            assert encoder_final_state.get_shape().as_list() == [batch_size, cell_depth]

            cell = tf.contrib.rnn.BasicRNNCell(cell_depth)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state=encoder_final_state)
            outputs, state = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)

            self.assertIsInstance(outputs, tf.contrib.seq2seq.BasicDecoderOutput)
            self.assertIsInstance(state, tf.Tensor)

            sess.run(tf.global_variables_initializer())
            outputs_result, state_result = sess.run([outputs, state])

            self.assertIsInstance(outputs_result, tf.contrib.seq2seq.BasicDecoderOutput)
            self.assertEqual(outputs_result.rnn_output.dtype, np.float32)
            self.assertEqual(outputs_result.sample_id.dtype, np.int32)
            self.assertEqual(outputs_result.rnn_output.shape[0], batch_size)
            self.assertEqual(outputs_result.sample_id.shape[0], batch_size)
            self.assertEqual(outputs_result.rnn_output.shape[1], outputs_result.sample_id.shape[1])
            self.assertEqual(outputs_result.rnn_output.shape[2], cell_depth)

            iterations = outputs_result.sample_id.shape[1]
            self.assertLessEqual(iterations, maximum_iterations)
            if iterations < maximum_iterations:
                self.assertEqual(outputs_result.sample_id[0][-1], end_token)

            self.assertEqual(state_result.shape, (1, 3))


if __name__ == '__main__':
    unittest.main()
