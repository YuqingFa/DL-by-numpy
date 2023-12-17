import numpy as np

a = np.zeros((2, 3))
b = np.full((2, 3), 1)

print(a)
print(b)

c = np.random.random((2, 3))
print(c)

d = np.random.randint(0, 10, size=(3, 2))
print(d)

e = [[2, 3, 4], [4, 3, 2]]
e = np.array(e)
print(e)

f = np.dot(d, b)    # 矩阵相乘
print(f)

g = np.array([1, 2])
h = np.array([2, 3])
i = np.dot(g, h)
print(i)

j = c + e
print(j)
k = c * e
print(k)

mean = np.mean(k, axis=0)
print(mean)
mean = np.mean(k)
print(mean)

Max = np.maximum(1, k)  # 可以两个数组
print(Max)
