if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import dezero.functions as F

from dezero import Function, Variable
from dezero.utils import plot_dot_graph
from dezero.utils import sum_to

np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)

# rng = np.random.default_rng(0)
# x = rng.random((100, 1))
# y = 5 + 2 * x + rng.random((100, 1))

# plt.scatter(x, y)
# plt.show()

x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

def predict(x):
    y = F.matmul(x, W) + b
    return y

def mean_squared_error_simple(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)

lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)

plt.scatter(x.data, y.data, c='steelblue', edgecolors='white', s=70)
plt.plot(x.data, predict(x).data, color='black', lw=2)
plt.xlabel('x')
plt.ylabel('y')
plt.show()