import numpy as np

class activation:
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
        output = activation.ReLU(output)
        return output

class Network:
    def __init__(self, network_shape):
        self.shape = network_shape
        self.layers = []
        for i in range(len(network_shape) - 1):
            layer = Layer(network_shape[i], network_shape[i + 1])
            self.layers.append(layer)

    def calculate_forword(self, inputs):
        outputs = [inputs]
        for i in range(len(self.shape) - 1):
            layer_output = self.layers[i].calculate_forward(outputs[i])
            outputs.append(layer_output)
        return outputs

def main():
    model = Network(network_shape)
    output = model.calculate_forword(a)
    print(output[-1])


if __name__ == "__main__":
    a = np.random.randn(5, 2)
    network_shape = [2, 128, 512, 2]

    main()
