if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dezero import Function, Variable, Parameter
from dezero import Model
from dezero.utils import plot_dot_graph
from dezero import optimizers
from dezero.models import MLP
import dezero.layers as L
import dezero.functions as F


np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iter = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
# optimizer = optimizers.SGD(lr)
optimizer = optimizers.MomentumSGD(lr)
optimizer.setup(model)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)
    model.cleargrads()
    loss.backward()

    optimizer.update()
    if i % 1000 == 0:
        print(f'Loss for epoch {i}:', loss.data)


plt.scatter(x, y, c='steelblue', edgecolors='white', s=70)
plt.scatter(x, model(x).data, color='red', s=90)
plt.xlabel('x')
plt.ylabel('y')
plt.show()