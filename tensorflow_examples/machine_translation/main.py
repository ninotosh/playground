from typing import Iterable, List, Tuple

import numpy as np
import tensorflow as tf


VOCABULARY_SIZE = 6


class Summary:
    def __init__(self, sess, logdir='logdir'):
        self.sess = sess
        self.writer = tf.summary.FileWriter(logdir, graph=sess.graph)
        tf.summary.merge_all()

    def write(self, scalar, name, global_step=None):
        with tf.name_scope('summaries'):
            summary = tf.summary.merge([tf.summary.scalar(name, scalar)])
            self.writer.add_summary(self.sess.run(summary), global_step)
            tf.summary.merge_all()


def translate(sentence: Iterable) -> np.ndarray:
    # TODO
    return VOCABULARY_SIZE - np.array([n for n in sentence])
    s = []
    for n in sentence:
        if n % 10 == 0:
            s.append(n)
            s.append(n + 1)
        elif n % 10 == 1:
            continue
        else:
            s.append(n)

    return VOCABULARY_SIZE - np.array(s)


def pad(sequences: Iterable) -> List:
    """pads with 0s to align dimensions"""
    result = []
    max_sentence_len = max(map(len, sequences))
    for sequence in sequences:
        result.append(
            np.lib.pad(
                sequence,
                pad_width=(0, max_sentence_len - len(sequence)),
                mode='constant',
                constant_values=0
            ).tolist()
        )

    return result


def one_hot(sequences: Iterable) -> List:
    result = []
    for sequence in sequences:
        sequence_result = []
        for element in sequence:
            zeros = np.zeros(VOCABULARY_SIZE, dtype=np.int)
            if element != 0:
                zeros.put(element, 1)

            sequence_result.append(zeros.tolist())

        result.append(sequence_result)

    return result


def create_training_data(training_data_size: int) -> Tuple:
    min_sentence_len = 3
    max_sentence_len = 5

    sources = []
    targets = []
    while len(targets) < training_data_size:
        sentence_len = np.random.randint(
            low=min_sentence_len,
            high=max_sentence_len + 1
        )  # low <= randint < high
        source = np.random.randint(low=1, high=VOCABULARY_SIZE, size=sentence_len)
        target = translate(source.tolist())
        if len(target) > 0:
            sources.append(source)
            targets.append(np.concatenate((target, [0])))  # append <eos>

    return one_hot(pad(sources)), one_hot(pad(targets))


def main(_):
    epochs = 10
    training_data_size = 2
    cell_size = VOCABULARY_SIZE
    batch_size = 1
    allow_smaller_final_batch = True

    assert batch_size <= training_data_size

    input_data, label_data = create_training_data(training_data_size)
    training_data = tf.train.slice_input_producer(
        [input_data, label_data],
        num_epochs=epochs
    )
    inputs, labels = map(tf.to_float, tf.train.batch(
        training_data,
        batch_size,
        allow_smaller_final_batch=allow_smaller_final_batch
    ))

    assert inputs.shape.as_list() == [
        None if allow_smaller_final_batch else batch_size,
        np.array(input_data).shape[1],  # max len of input sequence
        VOCABULARY_SIZE
    ]
    assert labels.shape.as_list() == [
        None if allow_smaller_final_batch else batch_size,
        np.array(label_data).shape[1],  # max len of label sequence
        VOCABULARY_SIZE
    ]

    time_steps = tf.reduce_sum(tf.reduce_sum(inputs, axis=2), axis=1)
    ideal = tf.arg_max(labels, dimension=2)

    cell = tf.contrib.rnn.BasicLSTMCell(cell_size)
    with tf.variable_scope('encoder'):  # cell.__call__() creates vars
        embedded = tf.contrib.layers.fully_connected(
            inputs=inputs,
            num_outputs=cell_size,
            activation_fn=tf.tanh
        )
        logits, final_state_tuple = tf.nn.dynamic_rnn(
            cell,
            embedded,
            sequence_length=time_steps,
            dtype=tf.float32,
        )
    loss = decode_for_training(final_state_tuple.c, labels)
    prediction = tf.arg_max(logits, dimension=2)

    global_step = tf.get_variable(
        'global_step',
        shape=[],
        initializer=tf.constant_initializer(0),
        trainable=False
    )
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.1)
    train = optimizer.minimize(tf.reduce_sum(loss, axis=0), global_step=global_step)

    fetches = {
        'inputs': inputs,
        'labels': labels,
        'time_steps': time_steps,
        'ideal': ideal,
        'logits': logits,
        'prediction': prediction,
        'loss': loss,
        'train': train,
    }

    with tf.Session() as sess:
        sess.run(tf.group(
            tf.global_variables_initializer(),
            # initialize num_epochs for slice_input_producer()
            tf.local_variables_initializer()
        ))

        summary = Summary(sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                result = sess.run(fetches=fetches)
                summary.write(tf.reduce_mean(loss), 'mean loss', tf.train.global_step(sess, global_step))
                print(result['loss'], 'loss')
                # print(result['sum'], 'sum')
                # print(np.exp(result['sum']), 'perplexity')
        except tf.errors.OutOfRangeError:
            print('batch stopped')
        finally:
            coord.request_stop()

        coord.join(threads)


def decode_for_training(final_enc_state, labels):
    # [actual batch size, max seq len, decoder cell size]
    tf.assert_rank(labels, 3)

    cell_size = labels.get_shape().as_list()[2]
    context = tf.contrib.layers.fully_connected(
        inputs=final_enc_state,
        num_outputs=cell_size,
        activation_fn=tf.tanh
    )

    # [actual batch size, decoder cell size]
    assert context.get_shape().as_list() == [None, cell_size]

    # tf.shape(labels): tuple of 1 element
    batch_size = tf.shape(labels)[0]  # type: tf.Tensor of rank 0
    # labels = tf.concat([tf.reshape(context, [batch_size, 1, cell_size]), labels], axis=1)

    cell = tf.contrib.rnn.BasicLSTMCell(cell_size)
    with tf.variable_scope('decoder'):
        def cond(loop_cnt, output_one_hot, _, __):
            # output_one_hot = tf.Print(output_one_hot, [output_one_hot, tf.zeros_like(output_one_hot)], 'XXX', -1, 100)
            # limit the number of output length to avoid infinite generation
            is_regular_word = tf.reduce_any(tf.not_equal(
                output_one_hot,
                tf.zeros_like(output_one_hot)
            ))
            return is_regular_word

        def body(loop_cnt, prev_label, prev_state, losses):
            cell_output, state = cell(prev_label, prev_state)
            output = tf.contrib.layers.fully_connected(
                inputs=cell_output,
                num_outputs=cell_size,
                activation_fn=tf.tanh
            )

            # cut out the `loop_cnt`-th label
            label = tf.reshape(
                tf.slice(labels, begin=[0, loop_cnt, 0], size=[batch_size, 1, cell_size]),
                shape=[batch_size, cell_size]
            )

            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=output,
                labels=label
            )

            return (
                tf.add(loop_cnt, 1),
                label,
                state,
                losses.write(loop_cnt, loss)
            )

        _, _, _, losses = tf.while_loop(
            cond,
            body,
            loop_vars=(
                tf.constant(0),
                context,
                cell.zero_state(batch_size=batch_size, dtype=tf.float32),
                tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            ),
        )

        return losses.stack()


if __name__ == '__main__':
    tf.app.run()
