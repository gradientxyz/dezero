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
from dezero import as_variable


# x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# x = Variable(np.array([[0.2, -0.4]]))
model = MLP((10, 3))
# model = MLP((10, 3), activation=F.relu)
# y = F.get_item(x, 1)
# indices = np.array([0, 0, 1])
# y = F.get_item(x, indices)
# y = x[1]

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
x = as_variable(x)

y = model(x)
# p = F.softmax(y)

# print(p, p.sum(axis=1))

loss = F.softmax_cross_entropy_simple(y, t)
print(loss)
loss = F.softmax_cross_entropy(y, t)
print(loss)



# y = model(x)
# p = F.softmax_simple(y)
# print(y)
# print(p, p.sum(axis=1))