import unittest

import numpy as np
import tensorflow as tf


class TestSequenceExample(unittest.TestCase):
    def test_parse_single_sequence_example(self):
        serialized = self.__class__.make_example([1, 2]).SerializeToString()
        context_features = {
            'length': tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            'mod_2': tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }
        context_output, feature_list_output = tf.parse_single_sequence_example(
            serialized,
            context_features=context_features,
            sequence_features=sequence_features
        )

        sess = tf.Session()
        context, feature_list = sess.run([context_output, feature_list_output])

        self.assertDictEqual(context, {'length': 2})
        np.testing.assert_equal(feature_list, {'mod_2': np.array([1, 0])})

    @staticmethod
    def make_example(sequence):
        example = tf.train.SequenceExample()
        example.context.feature['length'].int64_list.value.append(len(sequence))
        mod_2 = example.feature_lists.feature_list['mod_2']
        for element in sequence:
            mod_2.feature.add().int64_list.value.append(element % 2)

        return example


if __name__ == '__main__':
    unittest.main()
