import time
from typing import Iterable, List, Tuple

import numpy as np
import tensorflow as tf

VOCABULARY_SIZE = 10
FEATURE_SIZE = VOCABULARY_SIZE + 1


def translate(sentence: Iterable) -> np.ndarray:
    simple = False
    if simple:
        return FEATURE_SIZE - np.array([n for n in sentence])

    s = []
    for n in sentence:
        if n % 5 == 0:
            s.append(n)
            s.append(n + 1)
        elif n % 5 == 1:
            continue
        else:
            s.append(n)

    return FEATURE_SIZE + 1 - np.array(s)


def pad(sequences: Iterable) -> List:
    """pads with -1s to align dimensions
    e.g.
        [
            [1, 2, 1],
            [2, 0]
        ]
    ->
        [
            [1, 2, 1],
            [2, 0, -1]
        ]
    """
    result = []
    max_sequence_len = max(map(len, sequences))
    for sequence in sequences:
        result.append(
            np.lib.pad(
                sequence,
                pad_width=(0, max_sequence_len - len(sequence)),
                mode='constant',
                constant_values=-1
            ).tolist()
        )

    for sequence in result:
        assert len(sequence) == max_sequence_len

    return result


def one_hot(sequences: Iterable) -> List:
    """e.g.
        [
            [1, 2, 1],
            [2, 0, -1]
        ]
    ->
        [
            [[0, 1, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 0, 0]]
        ]
    """
    result = []
    for sequence in sequences:
        sequence_result = []
        for element in sequence:
            zeros = np.zeros(FEATURE_SIZE, dtype=np.int)
            if element >= 0:
                zeros.put(element, 1)

            sequence_result.append(zeros.tolist())

        result.append(sequence_result)

    return result


def create_training_data(training_data_size: int) -> Tuple:
    min_sentence_len = 4
    max_sentence_len = 7
    assert 1 <= min_sentence_len <= max_sentence_len

    sources = []
    targets = []
    while len(targets) < training_data_size:
        sentence_len = np.random.randint(
            low=min_sentence_len,
            high=max_sentence_len + 1
        )  # low <= randint < high
        # 0 is reserved for <eos>
        source = np.random.randint(low=1, high=FEATURE_SIZE, size=sentence_len)
        target = translate(source.tolist())
        if len(target) > 0:
            sources.append(source)
            targets.append(np.concatenate((target, [0])))  # append <eos>

    return one_hot(pad(sources)), one_hot(pad(targets))


def main(_):
    epochs = 30000
    # out of VOCABULARY_SIZE ** max_sentence_len patterns
    training_data_size = 20
    encoder_cell_size = max(FEATURE_SIZE, 5)
    batch_size = 20
    total_steps = epochs * (training_data_size / batch_size)
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
        np.array(input_data).shape[1],  # max len of input sequences
        FEATURE_SIZE
    ]
    assert labels.shape.as_list() == [
        None if allow_smaller_final_batch else batch_size,
        np.array(label_data).shape[1],  # max len of label sequences
        FEATURE_SIZE
    ]

    encoder_cell = tf.contrib.rnn.BasicLSTMCell(encoder_cell_size)
    final_state_tuple = encode(inputs, encoder_cell)

    decoder_cell_size = labels.get_shape().as_list()[2]
    decoder_cell = tf.contrib.rnn.BasicLSTMCell(decoder_cell_size)
    loss = decode_for_training(decoder_cell, final_state_tuple.c, labels)
    perplexity = tf.reduce_mean(tf.exp(loss))

    global_step = tf.get_variable(
        'global_step',
        shape=[],
        initializer=tf.constant_initializer(0),
        trainable=False
    )

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss, global_step=global_step)

    summary = tf.summary.merge([
        tf.summary.scalar('mean_perplexity', perplexity)
    ])

    fetches = {
        'inputs': inputs,
        'labels': labels,
        'loss': loss,
        'perplexity': perplexity,
        'train': train,
        'summary': summary
    }

    with tf.Session() as sess:
        sess.run(tf.group(
            tf.global_variables_initializer(),
            # initialize num_epochs for slice_input_producer()
            tf.local_variables_initializer()
        ))

        writer = tf.summary.FileWriter(
            'logdir/{}'.format(int(time.time())),
            graph=sess.graph
        )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                result = sess.run(fetches=fetches)
                step = tf.train.global_step(sess, global_step)
                writer.add_summary(result['summary'], step)
                if total_steps < 10 or step % (int(total_steps / 10)) == 0:
                    print('{}: {}'.format(step, result['perplexity']))
        except tf.errors.OutOfRangeError:
            print('batch ended')
        finally:
            coord.request_stop()

        coord.join(threads)

        correct, total = calculate_accuracy(
            sess, encoder_cell, decoder_cell, input_data, label_data)
        print('training accuracy: {} / {}'.format(correct, total))


def encode(inputs, cell, reuse=False):
    tf.assert_rank(inputs, 3)

    time_steps = tf.reduce_sum(tf.reduce_sum(inputs, axis=2), axis=1)

    with tf.variable_scope('encoder', reuse=reuse):
        embedded = tf.contrib.layers.fully_connected(
            inputs=inputs,
            num_outputs=cell.output_size,
            activation_fn=tf.sigmoid
        )

        tf.assert_rank(embedded, 3)

        _, final_state_tuple = tf.nn.dynamic_rnn(
            cell,
            embedded,
            sequence_length=time_steps,
            dtype=tf.float32,
        )

        return final_state_tuple


def bridge(final_enc_state, decoder_cell_size, reuse=False):
    tf.assert_rank(final_enc_state, 2)

    with tf.variable_scope('bridge', reuse=reuse):
        context = tf.contrib.layers.fully_connected(
            inputs=final_enc_state,
            num_outputs=decoder_cell_size,
            activation_fn=tf.tanh
        )

        return context


def decode_for_training(cell, final_enc_state, labels):
    # [actual batch size, max seq len, decoder cell size]
    tf.assert_rank(labels, 3)

    cell_size = cell.output_size
    context = bridge(final_enc_state, cell_size)

    # [actual batch size, decoder cell size]
    assert context.get_shape().as_list() == [None, cell_size]

    # tf.shape(labels): tuple of 1 element
    batch_size = tf.shape(labels)[0]  # type: tf.Tensor of rank 0
    max_time_step = labels.get_shape()[1].value

    with tf.variable_scope('decoder'):
        def cond(loop_cnt, _, __, ___):
            return tf.less(loop_cnt, max_time_step)

        def body(loop_cnt, prev_label, prev_state, losses):
            cell_output, state = cell(prev_label, prev_state)
            output = decoder_projection(cell_output, cell_size)

            # cut out the `loop_cnt`-th label
            label = tf.reshape(
                tf.slice(labels, begin=[0, loop_cnt, 0], size=[batch_size, 1, cell_size]),
                shape=[batch_size, cell_size]
            )

            # loss for output past the last time step is calculated to be 0
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=output,
                labels=label
            )

            return (
                tf.add(loop_cnt, 1),
                # pass the label as the output of the current step
                label,
                state,
                losses.write(loop_cnt, loss)
            )

        _, _, _, result_loss = tf.while_loop(
            cond,
            body,
            loop_vars=(
                tf.constant(0),
                context,
                cell.zero_state(batch_size=batch_size, dtype=tf.float32),
                tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            ),
        )

        losses = tf.reduce_sum(result_loss.stack(), axis=0)
        time_steps = tf.reduce_sum(tf.reduce_sum(labels, axis=2), axis=1)
        return tf.div(losses, time_steps)


def decoder_projection(input_, num_outputs, reuse=False):
    return input_  # pass through


def infer(encoder_cell, decoder_cell, sentences):
    tf.assert_rank(sentences, 3)
    assert sentences.get_shape()[0].value == 1  # batch size
    assert sentences.get_shape()[2].value == FEATURE_SIZE

    # stops generating output if the length reaches the double of the source
    output_len_threshold = sentences.get_shape()[1].value * 2

    final_state_tuple = encode(sentences, encoder_cell, reuse=True)
    context = bridge(final_state_tuple.c, decoder_cell.output_size, reuse=True)

    with tf.variable_scope('decoder', reuse=True):
        def cond(loop_cnt, prev_out, _, __):
            less = tf.less(loop_cnt, output_len_threshold)
            is_regular_word = tf.reduce_any(
                tf.not_equal(
                    prev_out,
                    tf.one_hot([0], FEATURE_SIZE)  # <eos>
                )
            )

            return tf.logical_and(less, is_regular_word)

        def body(loop_cnt, prev_out, prev_state, result):
            cell_output, state = decoder_cell(prev_out, prev_state)
            num_outputs = decoder_cell.output_size
            output = decoder_projection(
                cell_output,
                num_outputs=num_outputs,
                reuse=True
            )
            arg_max = tf.arg_max(output, dimension=1)
            one_hot_output = tf.one_hot(
                indices=arg_max,
                depth=num_outputs
            )

            return (
                tf.add(loop_cnt, 1),
                one_hot_output,
                state,
                result.write(result.size(), tf.cast(one_hot_output, dtype=tf.int8))
            )

        _, __, ___, inferred = tf.while_loop(
            cond,
            body,
            loop_vars=(
                tf.constant(0),
                context,
                decoder_cell.zero_state(batch_size=1, dtype=tf.float32),
                tf.TensorArray(tf.int8, size=0, dynamic_size=True)
            )
        )

        return inferred.stack()


def calculate_accuracy(sess, encoder_cell, decoder_cell, inputs, labels):
    test_results = []
    for i in range(len(inputs)):
        test_inputs = inputs[i:i+1]
        test_labels = labels[i:i+1]

        outputs = infer(
            encoder_cell,
            decoder_cell,
            tf.constant(test_inputs, dtype=tf.float32)
        )
        inferred = sess.run(outputs)
        inferred = inferred.reshape(
            inferred.shape[1],
            inferred.shape[0],
            inferred.shape[2]
        )

        # pad `inferred` with zeros
        test_label_len = np.array(test_labels).shape[1]
        inferred_len = inferred.shape[1]
        if inferred_len < test_label_len:
            inferred = np.concatenate(
                (inferred, np.zeros([1, test_label_len - inferred_len, FEATURE_SIZE])),
                axis=1
            )

        test_results.append(np.array_equal(inferred, test_labels))

    return np.count_nonzero(test_results), len(test_results)


if __name__ == '__main__':
    tf.app.run()
