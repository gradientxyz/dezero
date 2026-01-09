if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import dezero.functions as F

from dezero import Function, Variable
from dezero.utils import plot_dot_graph

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# plt.scatter(x, y, c='steelblue', edgecolors='white', s=70)
# plt.show()

# weights initialization
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))

# network prediction
def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y

lr = 0.2
iters = 10000

# train the network
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    if i % 1000 == 0:
        print(f'Loss for epoch {i}:', type(loss.data), loss.data)

print(f'Loss for epoch {i}:', type(loss.data), loss.data)

plt.scatter(x, y, c='steelblue', edgecolors='white', s=70)
plt.scatter(x, predict(x).data, color='red', s=90)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

