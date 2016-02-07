#!/usr/bin/env python
# -*- coding: utf-8 -*-
from random import uniform


def sign(z):
    """
    :type z: float
    :return: int
    """
    return 1 if z > 0 else -1


def subtract(target, output):
    """
    :type target: int
    :type output: int
    :return: float
    """
    return target - output


class Perceptron:
    def __init__(self, learning_rate, weights, bias, activate=sign):
        """
        :type learning_rate: float
        :type weights: [float]
        :type bias: float
        """
        assert 0 < learning_rate <= 1
        assert len(weights) > 0

        self.learning_rate = learning_rate
        self.weights = weights
        self.bias = bias
        self.activate = activate
        # max of (input norm)**2
        # should be max of all training samples,
        # but actually updated with each sample
        self.max_squared_norm = 0.0

    def _sum_up(self, inputs):
        """
        :type inputs: [float]
        :return: float
        """
        assert len(inputs) == len(self.weights)

        return sum([w * i for w, i in zip(self.weights, inputs)]) + self.bias

    def output(self, inputs):
        """
        :type inputs: [float]
        :return: float
        """
        assert len(inputs) == len(self.weights)

        z = self._sum_up(inputs)

        return self.activate(z)

    def _update(self, error, inputs):
        """
        :type error: float
        :type inputs: [float]
        """
        assert len(inputs) == len(self.weights)

        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * inputs[i]

        norm = sum([i ** 2 for i in inputs])
        self.max_squared_norm = max(self.max_squared_norm, norm)
        self.bias += self.learning_rate * error * 1 * self.max_squared_norm

    def train(self, inputs, target):
        """
        :type inputs: [float]
        :type target: int
        """
        assert len(inputs) == len(self.weights)

        out = self.output(inputs)
        error = subtract(target, out)

        if error == 0:
            return

        self._update(error, inputs)


def run(
        training_input_list,
        training_targets,
        test_input_list,
        learning_rate,
        initial_weights,
        initial_bias):
    assert len(training_input_list) > 0
    assert len(training_input_list) == len(training_targets)
    assert len(test_input_list) > 0
    assert len(test_input_list[0]) == len(training_input_list[0])
    assert 0 < learning_rate <= 1
    assert len(initial_weights) == len(training_input_list[0])

    perceptron = Perceptron(learning_rate, initial_weights, initial_bias)

    for i in range(len(training_input_list)):
        perceptron.train(
            training_input_list[i],
            training_targets[i]
        )

    print('weights learned: {}'.format(perceptron.weights))
    print('   bias learned: {:.3f}'.format(perceptron.bias))

    return [perceptron.output(test_inputs) for test_inputs in test_input_list]


def run_n_dimensional(
        n_training,
        learning_rate,
        initial_weights,
        initial_bias,
        generate_inputs,
        get_target):

    training_input_list = []
    training_targets = []
    for i in range(n_training):
        inputs = generate_inputs(i)
        training_input_list.append(inputs)
        training_targets.append(get_target(inputs))

    n_test = min(1000, n_training)
    test_input_list = []
    test_targets = []
    for i in range(n_test):
        inputs = generate_inputs(i)
        test_input_list.append(inputs)
        test_targets.append(get_target(inputs))

    test_outputs = run(
        training_input_list,
        training_targets,
        test_input_list,
        learning_rate,
        initial_weights,
        initial_bias
    )

    errors = 0
    for i in range(n_test):
        if test_outputs[i] != test_targets[i]:
            errors += 1

    print('error rate: {} / {} => {:.3f}%'.format(
        errors,
        n_test,
        100.0 * errors / n_test
    ))


def run_1d(learning_rate, initial_weights, initial_bias):
    print('\n{}({}, {}, {})'.format(
        run_1d.__name__, learning_rate, initial_weights, initial_bias
    ))

    intercept = uniform(0, 1)

    def generate_inputs(_):
        return [uniform(intercept - 1.0, intercept + 1.0)]

    def get_target(inputs):
        return 1 if inputs[0] < intercept else -1

    run_n_dimensional(
        10000,
        learning_rate,
        initial_weights,
        initial_bias,
        generate_inputs,
        get_target
    )


def run_and(learning_rate, initial_weights, initial_bias):
    print('\n{}({}, {}, {})'.format(
        run_and.__name__, learning_rate, initial_weights, initial_bias
    ))

    def generate_inputs(i):
        # try to generate data in a random order
        return [[0, 1], [0, 0], [1, 1], [1, 0]][i]

    def get_target(inputs):
        return 1 if inputs[0] * inputs[1] > 0 else -1

    run_n_dimensional(
        4,
        learning_rate,
        initial_weights,
        initial_bias,
        generate_inputs,
        get_target
    )


def run_or(learning_rate, initial_weights, initial_bias):
    print('\n{}({}, {}, {})'.format(
        run_or.__name__, learning_rate, initial_weights, initial_bias
    ))

    def generate_inputs(i):
        # try to generate data in a random order
        return [[0, 1], [0, 0], [1, 1], [1, 0]][i]

    def get_target(inputs):
        return 1 if inputs[0] + inputs[1] > 0 else -1

    run_n_dimensional(
        4,
        learning_rate,
        initial_weights,
        initial_bias,
        generate_inputs,
        get_target
    )


def run_2d(learning_rate, initial_weights, initial_bias):
    print('\n{}({}, {}, {})'.format(
        run_2d.__name__, learning_rate, initial_weights, initial_bias
    ))

    a = uniform(1, 2)
    b = uniform(100, 101)

    def generate_inputs(_):
        return [uniform(-5, 5), uniform(b - 5, b + 5)]

    def get_target(inputs):
        x = inputs[0]
        y = inputs[1]
        # perceptron's weights are perfect when
        # (1, a) and (weights[0], weights[1]) are at right angles.
        # i.e. 1 * weights[0] + a * weights[1] == 0
        # perceptron's bias is perfect when
        # bias == -b * weights[1] or
        # bias == b / a * weights[0]
        return 1 if y - (a * x + b) > 0 else -1

    run_n_dimensional(
        1000000,
        learning_rate,
        initial_weights,
        initial_bias,
        generate_inputs,
        get_target
    )

if __name__ == "__main__":
    run_1d(uniform(0, 1), [uniform(0, 1)], uniform(0, 1))
    run_and(uniform(0, 1), [uniform(-1, 1), uniform(-1, 1)], uniform(-1, 1))
    run_or(uniform(0, 1), [uniform(-1, 1), uniform(-1, 1)], uniform(-1, 1))
    run_2d(uniform(0, 1), [uniform(-1, 1), uniform(-1, 1)], uniform(-1, 1))
