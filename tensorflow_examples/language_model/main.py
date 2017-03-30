import numpy as np
import tensorflow as tf

# vocabulary
#  0     1  2   3   4 5
# [<eos> it was hot . ?]
INPUT_DATA = [
    [
        [0, 1, 0, 0, 0, 0],  # it
        [0, 0, 1, 0, 0, 0],  # was
        [0, 0, 0, 1, 0, 0],  # hot
        [0, 0, 0, 0, 1, 0],  # .
        [1, 0, 0, 0, 0, 0],  # <eos>
    ],
    [
        [0, 0, 1, 0, 0, 0],  # was
        [0, 1, 0, 0, 0, 0],  # it
        [0, 0, 0, 0, 0, 1],  # ?
        [1, 0, 0, 0, 0, 0],  # <eos>
        [0, 0, 0, 0, 0, 0],  # no input
    ]
]
# this is INPUT_DATA shifted by 1
LABEL_DATA = [
    INPUT_DATA[0][1:] + [[0] * len(INPUT_DATA[0][0])],
    INPUT_DATA[1][1:] + [[0] * len(INPUT_DATA[0][0])],
]


class Summary:
    def __init__(self, sess, logdir='logdir'):
        self.sess = sess
        self.writer = tf.summary.FileWriter(logdir, graph=sess.graph)
        tf.summary.merge_all()

    def write(self, scalar, name, global_step=None):
        summary = tf.summary.merge([tf.summary.scalar(name, scalar)])
        self.writer.add_summary(self.sess.run(summary), global_step)


class Evaluator:
    @staticmethod
    def batch_accuracy(labels, predictions):
        """calculates accuracy of a batch like tf.metrics.batch_accuracy()
        without words after <eos>

        :param labels: e.g. [[1, 5, 0, 0, 0], [2, 3, 4, 0, 0]]
        :type labels: np.ndarray|list|Iterable
        :type predictions: np.ndarray|list|Iterable
        """
        ideal_result = list(labels.tolist()) if isinstance(labels, np.ndarray) else labels
        predicted_result = list(predictions.tolist()) if isinstance(predictions, np.ndarray) else predictions
        actual_batch_size = len(ideal_result)
        correct_count = 0
        for i in range(actual_batch_size):
            if Evaluator.sequence_accurate(ideal_result[i], predicted_result[i]):
                correct_count += 1

        return float(correct_count) / actual_batch_size

    @staticmethod
    def sequence_accurate(label, prediction):
        """determines whether a predicted sequence is accurate

        :param label: e.g. [2, 3, 4, 0, 0]
        :type label: Iterable
        :type prediction: Iterable
        :rtype: bool
        """
        end_index = label.index(0) + 1
        return label[:end_index] == prediction[:end_index]


def main(_):
    epochs = 200
    last_time_step = len(INPUT_DATA[0])
    vocabulary_size = len(INPUT_DATA[0][0])
    # these sizes should be equal because of no projection layers
    # between the RNN layer and the softmax layer
    cell_size = vocabulary_size
    batch_size = 2
    allow_smaller_final_batch = False

    training_data = tf.train.slice_input_producer(
        [INPUT_DATA, LABEL_DATA],
        num_epochs=epochs
    )
    inputs, labels = map(tf.to_float, tf.train.batch(
        training_data,
        batch_size,
        allow_smaller_final_batch=allow_smaller_final_batch
    ))

    assert inputs.shape.as_list() == [
        None if allow_smaller_final_batch else batch_size,
        last_time_step,
        vocabulary_size
    ]

    time_steps = tf.reduce_sum(tf.reduce_sum(inputs, axis=2), axis=1)
    ideal = tf.arg_max(labels, 2)

    cell = tf.contrib.rnn.BasicRNNCell(cell_size)
    with tf.variable_scope('my_rnn'):  # cell.__call__() creates vars
        logits, final_state = tf.nn.dynamic_rnn(
            cell,
            inputs,
            sequence_length=time_steps,
            dtype=tf.float32,
        )
    prediction = tf.arg_max(logits, 2)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits
    ))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)

    fetches = {
        'inputs': inputs,
        'labels': labels,
        'time_steps': time_steps,
        'ideal': ideal,
        'logits': logits,
        'prediction': prediction,
        'loss': loss,
        'train': train
    }

    with tf.Session() as sess:
        sess.run(tf.group(
            tf.global_variables_initializer(),
            # initialize num_epochs for slice_input_producer()
            tf.local_variables_initializer()
        ))
        _ = Summary(sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 0
        try:
            while not coord.should_stop():
                result = sess.run(fetches=fetches)
                accuracy = Evaluator.batch_accuracy(result['ideal'], result['prediction'])

                print('perplexity: {:.3f}, batch_accuracy: {}'.format(
                    np.exp(result['loss']),
                    accuracy
                ))

                if accuracy == 1.0:
                    break

                step += 1
                print('step {}'.format(step))
        except tf.errors.OutOfRangeError:
            print('batch stopped at step {}'.format(step))
        finally:
            coord.request_stop()

        coord.join(threads)

        inferred = []
        for sequence in INPUT_DATA:
            first_word = tf.constant([sequence[0]], tf.float32)
            # generate output from the 1st word
            inferred.append(sess.run(tf.arg_max(infer(cell, vocabulary_size, first_word), 1)).tolist())

        print('inference accuracy: {}'.format(
            Evaluator.batch_accuracy(np.argmax(LABEL_DATA, 2), inferred)
        ))


def infer(cell, vocabulary_size, first_one_hot):
    assert first_one_hot.shape.as_list() == [1, 6]

    with tf.variable_scope('my_rnn/rnn', reuse=True):
        def cond(output_one_hot, _, all_outputs):
            # limit the number of output length to avoid infinite generation
            less = tf.less(all_outputs.size(), 10)
            is_regular_word = tf.reduce_any(tf.not_equal(
                output_one_hot,
                tf.one_hot([[0]], vocabulary_size)  # <eos>
            ))
            return tf.reduce_all([less, is_regular_word])

        def body(input_one_hot, state, all_outputs):
            output, new_state = cell(input_one_hot, state)
            output_one_hot = tf.one_hot(
                tf.arg_max(output, dimension=1),
                vocabulary_size
            )
            return (
                output_one_hot,
                new_state,
                all_outputs.write(all_outputs.size(), output_one_hot)
            )

        _, _, inferred = tf.while_loop(
            cond,
            body,
            (
                first_one_hot,
                cell.zero_state(1, tf.float32),
                tf.TensorArray(tf.float32, size=0, dynamic_size=True),
            ),
        )

        return tf.reshape(inferred.stack(), [inferred.size(), vocabulary_size])
        # return this to include the first word
        # tf.concat([
        #     first_one_hot,
        #     tf.reshape(inferred.stack(), [inferred.size(), vocabulary_size])
        # ], axis=0)


if __name__ == '__main__':
    tf.app.run()
