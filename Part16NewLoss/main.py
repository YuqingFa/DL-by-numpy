import numpy as np
from data import create_data, plot_data
import copy

NETWORK_SHAPE = [2, 32, 2]
NUM_OF_DATA = 100
BATCH_SIZE = 100
LEARNING_RATE = 0.01

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

def vector_normalize(array):
    max_number = np.max(np.absolute(array))
    scale_rate = np.where(max_number == 0, 0, 1 / max_number)
    norm = array * scale_rate
    return norm

def create_weights(n_inputs, n_neurons):
    return np.random.randn(n_inputs, n_neurons)

def create_biases(n_neurons):
    return np.random.randn(n_neurons)

def classify(probabilities):
    return np.rint(probabilities[:, 1])

def precise_loss_function(predicted, real):
    real_matrix = np.zeros((len(real), 2))
    real_matrix[:, 1] = real
    real_matrix[:, 0] = 1 - real
    product = np.sum(predicted * real_matrix, axis=1)
    return 1 - product

def loss_function(predicted, real):
    condition = (predicted > 0.5)
    binary_predicted = np.where(condition, 1, 0)

    real_matrix = np.zeros((len(real), 2))
    real_matrix[:, 1] = real
    real_matrix[:, 0] = 1 - real
    product = np.sum(binary_predicted * real_matrix, axis=1)
    return 1 - product

def get_final_preAct_demands(predicted_values, target_vector):
    target = np.zeros((len(target_vector), 2))
    target[:, 1] = target_vector
    target[:, 0] = 1 - target_vector
    for i in range(len(target_vector)):
        if np.dot(target[i], predicted_values[i]) > 0.5:
            target[i] = np.array([0, 0])
        else:
            target[i] = (target[i] - 0.5) * 2
    return target

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weight = create_weights(n_inputs, n_neurons)
        self.biases = create_biases(n_neurons)

    def calculate_forward(self, inputs):
        output = np.dot(inputs, self.weight) + self.biases
        # output = activation.ReLU(output)
        return output

    def layer_backward(self, preWeights_values, afterWeights_demands):
        preWeights_demands = np.dot(afterWeights_demands, self.weight.T)

        condition = (preWeights_values > 0)
        value_derivatives = np.where(condition, 1, 0)
        preActs_demands = value_derivatives * preWeights_demands
        norm_preActs_demands = normalize(preActs_demands)

        weight_adjust_matrix = self.get_weight_adjust_matrix(preWeights_values, afterWeights_demands)
        norm_weight_adjust_matrix = normalize(weight_adjust_matrix)

        return (norm_preActs_demands, norm_weight_adjust_matrix)

    def get_weight_adjust_matrix(self, preWeight_values, afterWeights_demand):
        plain_weights = np.full(self.weight.shape, 1)
        weights_adjust_matrix = np.full(self.weight.shape, 0.)
        plain_weights_T = plain_weights.T

        for i in range(BATCH_SIZE):
            weights_adjust_matrix += (plain_weights_T * preWeight_values[i, :]).T * afterWeights_demand[i, :]
        weights_adjust_matrix /= BATCH_SIZE

        return weights_adjust_matrix

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

    def network_backward(self, layer_outputs, target_vector):
        backup_network = copy.deepcopy(self)
        preAct_demands = get_final_preAct_demands(layer_outputs[-1], target_vector)

        for i in range(len(self.layers)):
            layer = backup_network.layers[len(self.layers) - (i + 1)]
            if i != 0:
                layer.biases += LEARNING_RATE * np.mean(preAct_demands, axis=0)
                layer.biases = vector_normalize(layer.biases)

            outputs = layer_outputs[len(layer_outputs) - (2 + i)]
            results_list = layer.layer_backward(outputs, preAct_demands)
            preAct_demands = results_list[0]
            weight_adjust_matrix = results_list[1]
            layer.weight += LEARNING_RATE * weight_adjust_matrix
            layer.weight = normalize(layer.weight)

        return backup_network

    def one_batch_train(self, batch):
        dataset_X, dataset_y = batch[:, :2], copy.deepcopy(batch[:, 2]).astype(int)
        outputs = self.calculate_forword(dataset_X)
        precise_loss = precise_loss_function(outputs[-1], dataset_y)
        loss = loss_function(outputs[-1], dataset_y)

        if np.mean(precise_loss) <= 0.1:
            print("No need for training")
        else:

            backup_network = self.network_backward(outputs, dataset_y)
            backup_outputs = backup_network.calculate_forword(dataset_X)
            backup_precise_loss = precise_loss_function(backup_outputs[-1], dataset_y)
            backup_loss = loss_function(backup_outputs[-1], dataset_y)

            if np.mean(precise_loss) >= np.mean(backup_precise_loss) or np.mean(loss) >= np.mean(backup_loss):
                for i in range(len(self.layers)):
                    self.layers[i].weight = backup_network.layers[i].weight.copy()
                    self.layers[i].biases = backup_network.layers[i].biases.copy()
                print("improved")

            else:
                print("No Improvement")

        print("----------------------------------------")


def main():
    data = create_data(NUM_OF_DATA)
    plot_data(data, "true")

    model = Network(NETWORK_SHAPE)
    model.one_batch_train(data)


if __name__ == "__main__":
    main()
