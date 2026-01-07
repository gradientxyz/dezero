if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Function, Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F
import matplotlib.pyplot as plt


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)

x = Variable(np.random.randn(1, 2, 3))
y = x.reshape((2, 3))
print(y)
y = x.reshape(2, 3)
print(y)


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
print(x)
y = x.transpose()
print(y)
y = x.T
print(y)

x = np.random.randn(2, 3, 4)
print(x)
print('*' * 30)
y = x.transpose(1, 0, 2)
print(y)