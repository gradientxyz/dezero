if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dezero import Function, Variable, Parameter
from dezero import Model
from dezero.utils import plot_dot_graph
from dezero.models import MLP
import dezero.layers as L
import dezero.functions as F


# datasets
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# hyperparameters
lr = 0.2
max_iter = 10000
hidden_size = 10

# model
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y
    
# x = Variable(np.random.randn(5, 10), name='x')
# model = TwoLayerNet(100, 10)
# model.plot(x)

# model = TwoLayerNet(hidden_size, 1)
model = MLP((20, 10, 1))
# model.plot(x)

# train
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    
    if i % 1000 == 0:
        print(f'Loss for epoch {i}:', loss.data)


plt.scatter(x, y, c='steelblue', edgecolors='white', s=70)
plt.scatter(x, model(x).data, color='red', s=90)
plt.xlabel('x')
plt.ylabel('y')
plt.show()