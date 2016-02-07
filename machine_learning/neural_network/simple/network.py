import math

from simple.perceptron import Perceptron


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def derivative_sigmoid_written_in_itself(sig: float) -> float:
    """
    :param sig: output from sigmoid
    note: the derivative with respect to z == e ** z / (e ** z + 1) ** 2
    """
    return sig * (1 - sig)


def get_derivative_written_in_itself(f):
    if f == sigmoid:
        return derivative_sigmoid_written_in_itself

    return None


def squared_error(target: float, output: float) -> float:
    return 1.0 / 2.0 * (target - output) ** 2


def derivative_squared_error(target: float, output: float) -> float:
    return output - target


def squared_errors(targets: [float], outputs: [float]) -> float:
    assert len(targets) == len(outputs)

    return sum([squared_error(target, output) for target, output in zip(targets, outputs)])


class Layer:
    def __init__(self, perceptrons: [Perceptron]):
        self.perceptrons = perceptrons

    def output(self, inputs: [float]) -> [float]:
        return [perceptron.output(inputs) for perceptron in self.perceptrons]


class NetworkWith1HiddenLayer:
    def __init__(self, hidden_layer: Layer, output_layer: Layer, calc_error=squared_error):
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.calc_error = calc_error
        self.derivative_error = globals()['derivative_' + self.calc_error.__name__]

    def output(self, inputs: [float]) -> [float]:
        return self.output_layer.output(self.hidden_layer.output(inputs))

    def train(self, inputs: [float], target: [float]):
        hl_outputs = self.hidden_layer.output(inputs)
        ol_outputs = self.output_layer.output(hl_outputs)

        total_gradient_to_hl = [0.0 for _ in range(len(hl_outputs))]
        for i in range(len(ol_outputs)):
            ol_out = ol_outputs[i]
            perceptron = self.output_layer.perceptrons[i]
            derivative_activate = get_derivative_written_in_itself(perceptron.activate)
            delta = self.derivative_error(target[i], ol_out) * derivative_activate(ol_out)

            for j in range(len(hl_outputs)):
                hl_out = hl_outputs[j]
                total_gradient_to_hl[j] += delta * perceptron.weights[j]
                perceptron.weights[j] -= perceptron.learning_rate * delta * hl_out
                # print('h[{}] -> o[{}]: {}'.format(j, i, perceptron.weights[j]))

            # uncomment to update the biases
            # perceptron.bias -= perceptron.learning_rate * delta

        for j in range(len(total_gradient_to_hl)):
            hl_out = hl_outputs[j]
            hl_perceptron = self.hidden_layer.perceptrons[j]
            hl_derivative_activate = get_derivative_written_in_itself(hl_perceptron.activate)
            hl_delta = total_gradient_to_hl[j] * hl_derivative_activate(hl_out)
            for k in range(len(inputs)):
                hl_perceptron.weights[k] -= hl_perceptron.learning_rate * hl_delta * inputs[k]
                # print('i[{}] -> h[{}]: {}'.format(k, j, hl_perceptron.weights[k]))

            # uncomment to update the biases
            # hl_perceptron.bias -= hl_perceptron.learning_rate * hl_delta


def run_xor():
    """
    Simulates XOR.
    It is better to uncomment the code of updating the biases.
    """
    from random import uniform, randrange

    learning_rate = 1

    hidden_layer = Layer([
        Perceptron(learning_rate, [uniform(-1, 1), uniform(-1, 1)], uniform(-1, 1), activate=sigmoid),
        Perceptron(learning_rate, [uniform(-1, 1), uniform(-1, 1)], uniform(-1, 1), activate=sigmoid),
    ])

    output_layer = Layer([
        Perceptron(learning_rate, [uniform(-1, 1), uniform(-1, 1)], uniform(-1, 1), activate=sigmoid),
    ])

    network = NetworkWith1HiddenLayer(hidden_layer, output_layer)

    input_samples = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [[0], [1], [1], [0]]
    for _ in range(10000):
        i = randrange(0, len(input_samples))
        network.train(input_samples[i], targets[i])

    for i in range(len(input_samples)):
        print('{} => {}, target: {}'.format(input_samples[i], network.output(input_samples[i]), targets[i]))

if __name__ == "__main__":
    run_xor()
