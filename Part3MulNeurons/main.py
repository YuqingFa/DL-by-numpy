import numpy as np

np.random.seed(0)

def ReLU(x):
    return np.maximum(0, x)

def create_weights(n_input, n_neurons):
    return np.random.randn(n_input, n_neurons)

def create_biases(n_neurons):
    return np.random.randn((n_neurons))


a11 = 1
a12 = 4
a13 = 2
a21 = 2
a22 = 3
a23 = 1
a31 = 6
a32 = 3
a33 = 2

inputs = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

weights = create_weights(3, 2)
b = create_biases(2)

sum1 = np.dot(inputs, weights)
sum1 = sum1 + b
print(ReLU(sum1))
