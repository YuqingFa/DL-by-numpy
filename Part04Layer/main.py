import numpy as np


def ReLU(x):
    return np.maximum(0, x)

def create_weights(n_inputs, n_neurons):
    return np.random.randn(n_inputs, n_neurons)

def create_biases(n_neurons):
    return np.random.randn(n_neurons)

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weight = create_weights(n_inputs, n_neurons)
        self.biases = create_biases(n_neurons)

    def calculate_forward(self, inputs):
        output = np.dot(inputs, self.weight) + self.biases
        output = ReLU(output)
        return output


if __name__ == "__main__":
    a = np.random.randn(5, 2)
    list_neuron = [3, 4, 2]

    layer1 = Layer(a.shape[1], list_neuron[0])
    layer2 = Layer(list_neuron[0], list_neuron[1])
    layer3 = Layer(list_neuron[1], list_neuron[2])

    x = layer1.calculate_forward(a)
    x = layer2.calculate_forward(x)
    output = layer3.calculate_forward(x)

    print(output)