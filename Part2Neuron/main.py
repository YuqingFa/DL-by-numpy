import numpy as np

def ReLU(x):
    return np.maximum(0, x)


a1 = 0.9
a2 = 0.5
a3 = 0.7
inputs = np.array([a1, a2, a3])

w1 = 0.8
w2 = -0.4
w3 = 0
weights = np.array([w1, w2, w3]).reshape(-1, 1)

b1 = 1

sum1 = ReLU(np.dot(inputs, weights) + b1)
print(sum1)

