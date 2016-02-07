import unittest

from simple.perceptron import Perceptron
from simple.network import Layer, sigmoid, NetworkWith1HiddenLayer, squared_errors


class TestNetworkWith1HiddenLayer(unittest.TestCase):
    def test(self):
        learning_rate = 0.5

        hidden_layer = Layer([
            Perceptron(learning_rate, [0.15, 0.2], 0.35, activate=sigmoid),
            Perceptron(learning_rate, [0.25, 0.3], 0.35, activate=sigmoid)
        ])

        output_layer = Layer([
            Perceptron(learning_rate, [0.4, 0.45], 0.6, activate=sigmoid),
            Perceptron(learning_rate, [0.5, 0.55], 0.6, activate=sigmoid)
        ])

        network = NetworkWith1HiddenLayer(hidden_layer, output_layer)

        inputs = [0.05, 0.1]
        targets = [0.01, 0.99]

        for a, b in zip(network.output(inputs), [0.75136507, 0.772928465]):
            self.assertAlmostEqual(a, b)

        self.assertAlmostEqual(squared_errors(targets, network.output(inputs)), 0.29837110)

        network.train(inputs, targets)

        for a, b in zip(network.hidden_layer.perceptrons[0].weights, [0.149780716, 0.19956143]):
            self.assertAlmostEqual(a, b)

        for a, b in zip(network.hidden_layer.perceptrons[1].weights, [0.24975114, 0.29950229]):
            self.assertAlmostEqual(a, b)

        self.assertAlmostEqual(squared_errors(targets, network.output(inputs)), 0.291027774)

        for _ in range(10000 - 1):
            network.train(inputs, targets)

        for a, b in zip(network.output(inputs), [0.01591362044355068, 0.9840642735146238]):
            self.assertAlmostEqual(a, b)

        self.assertAlmostEqual(squared_errors(targets, network.output(inputs)), 0.000035102)

        self.assertEqual(network.hidden_layer.perceptrons[0].bias, 0.35)
        self.assertEqual(network.output_layer.perceptrons[0].bias, 0.6)
