if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import dezero.functions as F

from dezero import Function, Variable, Parameter
from dezero.utils import plot_dot_graph
import dezero.layers as L


# x = Variable(np.array(1.0))
# p = Parameter(np.array(2.0))
# y = x * p

# print(isinstance(p, Parameter))
# print(isinstance(x, Parameter))
# print(isinstance(y, Parameter))


# layer = Layer()
# layer.p1 = Parameter(np.array(1))
# layer.p2 = Parameter(np.array(2))
# layer.p3 = Variable(np.array(3))
# layer.p4 = 'test'

# print(layer._params)
# print('----------------')

# for name in layer._params:
#     print(name, layer.__dict__[name])

# datasets
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

l1 = L.Linear(10)
l2 = L.Linear(1)

def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y

lr = 0.2
iters = 10000

# train the network
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(f'Loss for epoch {i}:', loss.data)

print(f'Loss for epoch {i}:', loss.data)

plt.scatter(x, y, c='steelblue', edgecolors='white', s=70)
plt.scatter(x, predict(x).data, color='red', s=90)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
