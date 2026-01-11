if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import Function, Variable, Parameter
from dezero import Model
from dezero.utils import plot_dot_graph
from dezero import optimizers
from dezero.models import MLP
import dezero.layers as L
import dezero.functions as F
from dezero import as_variable
from dezero.datasets import Spiral
from dezero import DataLoader


# batch_size = 10
# max_epoch = 1

# train_set = Spiral(train=True)
# test_set = Spiral(train=False)
# train_loader = DataLoader(train_set, batch_size)
# test_loader = DataLoader(test_set, batch_size, shuffle=False)

# for epoch in range(max_epoch):
#     for x, t in train_loader:
#         print(x.shape, t.shape)
#         break

#     for x, t in test_loader:
#         print(x.shape, t.shape)
#         break


# hyperparameter
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# data and model
train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

# data_size = len(train_set)
# max_iter = math.ceil(data_size /batch_size)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for batch_x, batch_t in train_loader:
        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        acc = F.accuracy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data * len(batch_t))
        sum_acc += float(acc.data) * len(batch_t)
    
    if epoch % 10 == 0:
        print(f'epoch: {epoch + 1}\n \
              train loss: {sum_loss / len(train_set):.4f}, \
              accuracy: {sum_acc / len(train_set):.4f}')
    
    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    if epoch % 10 == 0:
        print(f' \
              test loss: {sum_loss / len(test_set):.4f}, \
              accuracy: {sum_acc / len(test_set):.4f}')