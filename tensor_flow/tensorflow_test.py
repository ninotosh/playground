import unittest

import numpy as np
import tensorflow as tf


class TestTensorflow(unittest.TestCase):
    def test_type_shape(self):
        x = tf.cast(np.array([[3, 2]]), tf.int32)
        w = tf.Variable([
            [1, 2, 3],
            [4, 5, 6]
        ])
        b = tf.Variable([[5, 6, 7]])
        z = tf.matmul(x, w) + b
        init = tf.initialize_all_variables()

        self.assertIsInstance(x, tf.Tensor)
        self.assertNotIsInstance(w, tf.Tensor)
        self.assertNotIsInstance(b, tf.Tensor)
        self.assertIsInstance(z, tf.Tensor)
        self.assertIsInstance(init, tf.Operation)

        self.assertListEqual(x.get_shape().as_list(), [1, 2])
        self.assertListEqual(w.get_shape().as_list(), [2, 3])
        self.assertListEqual(b.get_shape().as_list(), [1, 3])
        self.assertListEqual(z.get_shape().as_list(), [1, 3])

        with tf.Session() as sess:
            self.assertIsNone(sess.run(init))

            self.assertIsInstance(sess.run(z), np.ndarray)
            self.assertListEqual(sess.run(z).tolist(), [
                [3*1 + 2*4 + 5, 3*2 + 2*5 + 6, 3*3 + 2*6 + 7]
            ])

    def test_add_overwrite(self):
        p0 = tf.placeholder(tf.float32, [1, 2])
        p1 = tf.placeholder(tf.float32, [1, 2])
        plus = p0 + p1
        add = tf.add(p0, p1)

        with tf.Session() as sess:
            for i in range(10):
                v0 = sess.run(tf.truncated_normal(p0.get_shape()))
                v1 = sess.run(tf.truncated_normal(p1.get_shape()))

                _plus = sess.run(plus, feed_dict={p0: v0, p1: v1})
                _add = sess.run(add, feed_dict={p0: v0, p1: v1})

                self.assertListEqual(_plus.tolist(), _add.tolist())

    def test_placeholder(self):
        a = tf.placeholder(tf.int32, [1, 2])
        b = tf.constant([[3, 4]])
        c = a + b

        self.assertIsInstance(a, tf.Tensor)
        with tf.Session() as sess:
            self.assertListEqual(sess.run(c, feed_dict={a: [[2, 3]]}).tolist(), [
                [2 + 3, 3 + 4]
            ])
            self.assertListEqual(sess.run(c, feed_dict={a: [[3, 1]]}).tolist(), [
                [3 + 3, 1 + 4]
            ])

    def test_assign(self):
        x = tf.Variable([1, 2])

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            self.assertListEqual(sess.run(x).tolist(), [1, 2])

            assign_op = x.assign([3, 4])
            self.assertIsInstance(assign_op, tf.Tensor)
            self.assertFalse(assign_op == x)

            self.assertListEqual(sess.run(x).tolist(), [1, 2])
            self.assertListEqual(sess.run(assign_op).tolist(), [3, 4])
            self.assertListEqual(sess.run(x).tolist(), [3, 4])

            self.assertFalse(assign_op == x)
            self.assertListEqual(sess.run(tf.equal(assign_op, x)).tolist(), [True, True])

    def test_summary(self):
        a = tf.placeholder(tf.int32, [1, 2])
        b = tf.constant([
            [3],
            [4]
        ])
        c = tf.cast(tf.matmul(a, b), tf.float32)
        self.assertIsInstance(c, tf.Tensor)
        self.assertEqual(c.dtype, tf.float32)
        self.assertListEqual(
            c.get_shape().as_list(),
            [1, 1]
        )

        with tf.Session() as sess:
            writer = tf.train.SummaryWriter('/tmp/{}'.format(__name__), graph=sess.graph)
            merge = tf.merge_all_summaries()
            self.assertIsNone(merge)

            scalar_summary = tf.scalar_summary([['c']], c)
            self.assertIsInstance(scalar_summary, tf.Tensor)
            self.assertEqual(scalar_summary.dtype, tf.string)
            self.assertListEqual(scalar_summary.get_shape().as_list(), [])

            merge = tf.merge_summary([scalar_summary])
            self.assertIsInstance(merge, tf.Tensor)
            self.assertEqual(merge.dtype, tf.string)
            self.assertListEqual(merge.get_shape().as_list(), [])

            global_step = 0
            feed_dict = [[2, 3]]
            c_serialized = sess.run(merge, feed_dict={a: feed_dict})
            self.assertIsInstance(c_serialized, bytes)

            add_summary = writer.add_summary(c_serialized, global_step=global_step)
            self.assertIsNone(add_summary)

            global_step += 1
            feed_dict = [[3, 1]]
            c_serialized = sess.run(merge, feed_dict={a: feed_dict})
            writer.add_summary(c_serialized, global_step=global_step)


if __name__ == '__main__':
    unittest.main()
