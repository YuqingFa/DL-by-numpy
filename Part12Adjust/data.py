import numpy as np
import math
import random
import matplotlib.pyplot as plt

NUM_OF_DATA = 1000


def tag_entry(x, y):
    if math.sqrt(x ** 2 + y ** 2 < 1):
        tag = 0
    else:
        tag = 1

    return tag


def create_data(num_of_data):
    entry_list = []
    for i in range(num_of_data):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        tag = tag_entry(x, y)
        entry_list.append([x, y, tag])

    return np.array(entry_list)


def plot_data(data, title):
    color = []
    for i in data[:, 2]:
        if i == 0:
            color.append("orange")
        else:
            color.append("blue")

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=color)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    data = create_data(NUM_OF_DATA)
    plot_data(data, "plot")
