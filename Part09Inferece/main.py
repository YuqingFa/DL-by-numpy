import numpy as np
from data import create_data, plot_data

class activation:
    def ReLU(inputs):
        return np.maximum(0, inputs)

    def Sigmod(inputs):
        1 / (1 + np.exp(-inputs))

    def Softmax(inputs):
        max_values = np.max(inputs, axis=1, keepdims=True)
        slided_inputs = inputs - max_values
        exp_values = np.exp(slided_inputs)
        norm_base = np.sum(exp_values, axis=1, keepdims=True)
        norm_values = exp_values / norm_base
        return norm_values


def normalize(array):
    max_number = np.max(np.absolute(array), axis=1, keepdims=True)
    scale_rate = np.where(max_number == 0, 0, 1 / max_number)
    norm = array * scale_rate
    return norm

def create_weights(n_inputs, n_neurons):
    return np.random.randn(n_inputs, n_neurons)

def create_biases(n_neurons):
    return np.random.randn(n_neurons)

def classify(probabilities):
    return np.rint(probabilities[:, 1])

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weight = create_weights(n_inputs, n_neurons)
        self.biases = create_biases(n_neurons)

    def calculate_forward(self, inputs):
        output = np.dot(inputs, self.weight) + self.biases
        # output = activation.ReLU(output)
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
            layer_sum = self.layers[i].calculate_forward(outputs[i])
            if i < len(self.layers) - 1:
                layer_output = activation.ReLU(layer_sum)
                layer_output = normalize(layer_output)
            else:
                layer_output = activation.Softmax(layer_sum)
            outputs.append(layer_output)
        return outputs

def main():
    NUM_OF_DATA = 100
    data = create_data(NUM_OF_DATA)
    plot_data(data, "true")
    dataset_X, dataset_y = data[:, :2], data[:, 2]
    network_shape = [2, 32, 2]
    model = Network(network_shape)
    output = model.calculate_forword(dataset_X)

    result = classify(output[-1])
    data[:, 2] = result

    plot_data(data, "pred")

    return output


if __name__ == "__main__":
    main()
